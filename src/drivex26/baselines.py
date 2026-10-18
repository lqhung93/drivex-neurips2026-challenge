"""Kinematic floors H0a (constant velocity) and H0b (constant acceleration).

Inputs are anchor-frame history (N, 4, 2) and its timestamps relative to the anchor (N, 4)
in seconds; the anchor itself is the origin at t = 0. Future times are the nominal 2 Hz grid.
"""

import numpy as np

FUTURE_T = 0.5 * np.arange(1, 13)


def constant_velocity(hist_xy: np.ndarray, hist_t: np.ndarray) -> np.ndarray:
    v = -hist_xy[:, -1] / -hist_t[:, -1:]  # (anchor - last history) / dt
    return v[:, None, :] * FUTURE_T[None, :, None]


def constant_acceleration(hist_xy: np.ndarray, hist_t: np.ndarray) -> np.ndarray:
    # Finite-difference velocities are exact at interval midpoints for a quadratic path.
    t1, t2 = hist_t[:, -2:-1], hist_t[:, -1:]
    v_late = -hist_xy[:, -1] / -t2
    v_early = (hist_xy[:, -1] - hist_xy[:, -2]) / (t2 - t1)
    mid_late, mid_early = t2 / 2, (t1 + t2) / 2
    acc = (v_late - v_early) / (mid_late - mid_early)
    v0 = v_late - acc * mid_late
    t = FUTURE_T[None, :, None]
    return v0[:, None, :] * t + 0.5 * acc[:, None, :] * t ** 2
