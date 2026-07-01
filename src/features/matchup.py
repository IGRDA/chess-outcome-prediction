"""Underdog / matchup-shape family: who is favored and how the stage weights it.

Expert-knowledge features that reshape the raw rating edge by tournament stage
(a rating edge in the last round may matter differently than in round 1). All
inputs (``white_rating``, ``black_rating``, ``round`` and the per-event max
round) are known before the game, so this family is leakage-safe. It is kept
deliberately small; the full "pressure" zoo hurt holdout performance in prior
work and is left for the modeling stage to reconsider.
"""

from __future__ import annotations

import pandas as pd

LATE_ROUND_FRACTION = 0.75


def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add underdog and stage-scaled matchup-shape features."""
    result = df.copy()
    rating_diff = result["white_rating"] - result["black_rating"]
    max_round = result.groupby("event")["round"].transform("max")
    round_norm = result["round"] / max_round

    result["white_is_favorite"] = (rating_diff > 0).astype("int64")
    result["favorite_magnitude"] = rating_diff.abs()
    result["round_norm"] = round_norm
    result["rounds_remaining"] = (max_round - result["round"]).astype("int64")
    result["rating_edge_scaled_by_round"] = rating_diff * round_norm
    result["is_late_round"] = (round_norm >= LATE_ROUND_FRACTION).astype("int64")
    return result
