"""Assign full or reduced scores using team-level positional rankings."""

import math
from collections.abc import Mapping

import numpy as np
import polars as pl

STARTER_COUNTS = {
    "QB": 1,
    "RB": 3,
    "WR": 4,
    "TE": 2,
    "K": 1,
    "DST": 1,
}
BACKUP_SCORE_MULTIPLIER = 0.5
RANK_COLUMN = "feature_fantasypros_position_rank"
GROUP_COLUMNS = ("season", "week", "team", "position")


def apply_starter_rule(
    frame: pl.DataFrame,
    conditional_scores: np.ndarray,
    starter_counts: Mapping[str, int] = STARTER_COUNTS,
    backup_multiplier: float = BACKUP_SCORE_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply team-position starter slots and return scores and role labels."""
    required = {
        *GROUP_COLUMNS,
        "player_id",
        RANK_COLUMN,
        "feature_is_bye_week",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Starter assignment is missing columns: "
            + ", ".join(sorted(missing))
        )
    if not 0.0 <= backup_multiplier <= 1.0:
        raise ValueError("The backup score multiplier must be between 0 and 1.")

    scores = np.asarray(conditional_scores, dtype=float).copy()
    if scores.shape != (frame.height,) or not np.isfinite(scores).all():
        raise ValueError("Conditional scores must be finite and match the rows.")

    player_rows = frame.select(
        *GROUP_COLUMNS,
        "player_id",
        RANK_COLUMN,
    ).to_dicts()
    groups: dict[tuple, list[int]] = {}
    for index, row in enumerate(player_rows):
        position = row["position"]
        if position not in starter_counts:
            raise ValueError(f"No starter-slot rule is defined for {position}.")
        key = tuple(row[column] for column in GROUP_COLUMNS)
        groups.setdefault(key, []).append(index)

    starters = np.zeros(frame.height, dtype=bool)
    multipliers = np.full(frame.height, backup_multiplier, dtype=float)
    for group, indices in groups.items():
        position = group[-1]

        def sort_key(index: int) -> tuple:
            row = player_rows[index]
            rank = row[RANK_COLUMN]
            if rank is not None and math.isfinite(float(rank)):
                return (0, float(rank), -scores[index], str(row["player_id"]))
            return (1, 0.0, -scores[index], str(row["player_id"]))

        selected = sorted(indices, key=sort_key)[: starter_counts[position]]
        starters[selected] = True
        multipliers[selected] = 1.0

    positions = np.asarray(frame["position"].to_list(), dtype=object)
    quarterbacks = positions == "QB"
    scores[quarterbacks] = np.maximum(scores[quarterbacks], 0.0)
    bye_weeks = frame["feature_is_bye_week"].fill_null(False).to_numpy()
    scores[bye_weeks] = 0.0
    weighted_scores = scores * multipliers
    weighted_scores[bye_weeks] = 0.0
    return weighted_scores, scores, starters, multipliers
