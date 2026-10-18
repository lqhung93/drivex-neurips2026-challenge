"""doScenes instruction text: annotation CSV reader and a keyword maneuver tagger.

The reader mirrors the official loader (column names strip + lower, blank instructions
dropped, `Scene Number` -> `scene-%04d`) so counts match the official loader (2,450 non blank pairs over 732 scenes).
The tagger is a transparent regex baseline for EDA and per-maneuver breakdowns, not a model input.
"""

import re
from pathlib import Path

import pandas as pd

# Multi-label: one instruction can carry several tags ("turn left, then change to the right lane").
MANEUVER_PATTERNS: dict[str, str] = {
    "u_turn": r"\bu[- ]?turn",
    "turn_left": r"\b(turn(s|ing)?|take|make|hang)\s+(a\s+|the\s+)?((first|second|next|1st|2nd)\s+)?left(?!\s+lane)\b|\bleft[- ]turn",
    "turn_right": r"\b(turn(s|ing)?|take|make|hang)\s+(a\s+|the\s+)?((first|second|next|1st|2nd)\s+)?right(?!\s+lane)\b|\bright[- ]turn",
    "lane_change": r"\b(change|switch|merge|shift|slide|move|get)\b[^.;]*\blanes?\b|\blane change",
    "nudge": r"\b(avoid|around|swerve|go around|pass(ing)? the (parked|stopped))\b",
    "stop": r"\bstop(?!light|sign| ?light| ?sign)\b|\bstopped\b|\bwait\b|\byield\b|\bpark\b|\bpull over\b",
    "slow": r"\bslow(ly| down)?\b|\bdecelerat|\bcautious",
    "straight": r"\bstraight\b|\bcontinue\b|\bfollow the (road|lane|street)\b|\bkeep (going|driving)\b",
    "follow": r"\bfollow (the|that)\s+(\w+\s+){0,2}(car|vehicle|truck|bus|van|suv)\b|\bbehind the\b",
}
_COMPILED = {k: re.compile(v) for k, v in MANEUVER_PATTERNS.items()}


def maneuver_tags(text: str) -> list[str]:
    """Keyword tags in MANEUVER_PATTERNS order; empty list if nothing matches."""
    t = " ".join(str(text).lower().split())
    return [k for k, rx in _COMPILED.items() if rx.search(t)]


def load_annotations(annotation_dir) -> pd.DataFrame:
    """All CSVs under `annotation_dir`, non-blank instructions only.

    Columns: scene_name, instruction, instruction_type, has_static, has_dynamic, annotator_file.
    """
    frames = []
    for path in sorted(Path(annotation_dir).glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df["annotator_file"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["scene number"] = pd.to_numeric(df["scene number"], errors="coerce")
    df = df.dropna(subset=["scene number"])
    df["instruction"] = df["instruction"].fillna("").astype(str).str.strip()
    df = df[df["instruction"] != ""].copy()
    code = df["instruction type"].fillna("").astype(str).str.strip().str.lower()
    return pd.DataFrame({
        "scene_name": df["scene number"].map(lambda n: f"scene-{int(n):04d}"),
        "instruction": df["instruction"],
        "instruction_type": code,
        "has_static": code.str.contains("s"),
        "has_dynamic": code.str.contains("d"),
        "annotator_file": df["annotator_file"],
    }).reset_index(drop=True)
