"""Map encoder v1: a fixed vector of lane geometry around the anchor, nuScenes map expansion.

For the anchor pose, take the lane or lane connector closest in position and heading, follow it
and its outgoing lanes for up to 40 m, and describe the branches in the anchor frame (metres,
x forward, y left). Uses the devkit's NuScenesMap (get_records_in_radius, discretize_lanes,
get_outgoing_lane_ids); computed once per split and stored like the text features.
"""

from __future__ import annotations

import numpy as np

from drivex26.data import to_anchor_frame

NAMES = ("has_lane", "lateral_offset", "heading_offset",
         "main_x10", "main_y10", "main_x20", "main_y20", "main_x30", "main_y30", "main_x40", "main_y40",
         "left_x30", "left_y30", "right_x30", "right_y30", "n_branches", "dist_to_branch")
DIM = len(NAMES)
STEP = 0.5  # m between discretised lane poses
HORIZON = 40.0  # m followed along the lane graph
SCALE = 20.0  # metres are divided by this, like the history features of H1


def _wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def _closest_lane(nmap, xy, yaw, radius: float = 5.0):
    """(token, index of closest pose, poses (N, 3)) of the best lane, or None.

    Score = distance + 5 m per radian of heading difference; lanes pointing against the ego
    heading (more than 90 degrees off) are ignored.
    """
    records = nmap.get_records_in_radius(xy[0], xy[1], radius, ["lane", "lane_connector"])
    tokens = records["lane"] + records["lane_connector"]
    best = None
    for token, poses in nmap.discretize_lanes(tokens, STEP).items():
        poses = np.asarray(poses, dtype=np.float64)
        if not len(poses):
            continue
        dist = np.linalg.norm(poses[:, :2] - xy, axis=1)
        i = int(np.argmin(dist))
        dheading = abs(float(_wrap(poses[i, 2] - yaw)))
        if dheading > np.pi / 2:
            continue
        score = dist[i] + 5.0 * dheading
        if best is None or score < best[0]:
            best = (score, token, i, poses)
    return None if best is None else best[1:]


def _branches(nmap, token, start, poses, max_branches: int = 8) -> list[np.ndarray]:
    """Every path along the lane graph from poses[start], cut at HORIZON metres; list of (M, 3)."""
    out = []
    stack = [(poses[start:], token, 0)]
    while stack and len(out) < max_branches:
        path, last, depth = stack.pop()
        length = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum() if len(path) > 1 else 0.0
        nxt = nmap.get_outgoing_lane_ids(last) if depth < 4 else []
        if length >= HORIZON or not nxt:
            out.append(path)
            continue
        for lane in nxt:
            more = np.asarray(nmap.discretize_lanes([lane], STEP)[lane], dtype=np.float64)
            stack.append((np.concatenate([path, more[1:] if len(more) > 1 else more]), lane, depth + 1))
    return out


def _point_at(path_xy: np.ndarray, distance: float) -> np.ndarray:
    """Point at `distance` metres along a polyline, clamped at its end."""
    if len(path_xy) == 1:
        return path_xy[0]
    seg = np.linalg.norm(np.diff(path_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if distance >= cum[-1]:
        return path_xy[-1]
    j = int(np.searchsorted(cum, distance, side="right")) - 1
    t = (distance - cum[j]) / max(seg[j], 1e-9)
    return path_xy[j] + t * (path_xy[j + 1] - path_xy[j])


def lane_features(nmap, anchor_xy: np.ndarray, anchor_yaw: float) -> np.ndarray:
    """DIM floats for one anchor pose (global frame in, anchor frame out), scaled by SCALE."""
    xy = np.asarray(anchor_xy, dtype=np.float64)
    found = _closest_lane(nmap, xy, anchor_yaw)
    out = np.zeros(DIM, dtype=np.float32)
    if found is None:
        return out
    token, i, poses = found
    local = to_anchor_frame(poses[i:i + 1, :2], xy, anchor_yaw)[0]
    branches = [to_anchor_frame(b[:, :2], xy, anchor_yaw) for b in _branches(nmap, token, i, poses)]
    ends = np.stack([_point_at(b, 30.0) for b in branches])
    turn = np.array([abs(np.arctan2(e[1], max(e[0], 1e-3))) for e in ends])
    main = branches[int(np.argmin(turn))]
    lane_left = (len(poses) - 1 - i) * STEP
    values = [1.0, local[1] / SCALE, float(_wrap(poses[i, 2] - anchor_yaw)),
              *(_point_at(main, d) / SCALE for d in (10.0, 20.0, 30.0, 40.0)),
              ends[int(np.argmax(ends[:, 1]))] / SCALE, ends[int(np.argmin(ends[:, 1]))] / SCALE,
              len(branches) / 4.0, min(lane_left, HORIZON) / SCALE]
    out[:] = np.concatenate([np.atleast_1d(np.asarray(v, dtype=np.float64)) for v in values])
    return out
