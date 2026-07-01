"""Rating / base family: pre-game rating edge and Elo expectation.

Captures the single strongest signal in the problem — the current-game rating
difference — plus the standard Elo expected score. Every input
(``white_rating``, ``black_rating``) is on the game object and is an as-of-game
value, so this family is leakage-safe with no cold-start special-casing.
"""

from __future__ import annotations

import pandas as pd


def elo_expected_score(white_rating: pd.Series, black_rating: pd.Series) -> pd.Series:
    """White's Elo expected score against black (0..1)."""
    return 1.0 / (1.0 + 10.0 ** ((black_rating - white_rating) / 400.0))


def add_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rating-difference and Elo-expectation features."""
    result = df.copy()
    rating_diff = result["white_rating"] - result["black_rating"]
    result["rating_diff"] = rating_diff
    result["abs_rating_diff"] = rating_diff.abs()
    result["avg_rating"] = (result["white_rating"] + result["black_rating"]) / 2.0
    result["white_expected_score"] = elo_expected_score(
        result["white_rating"], result["black_rating"]
    )
    result["black_expected_score"] = 1.0 - result["white_expected_score"]
    return result
