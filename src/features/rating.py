"""Rating / base family: pre-game rating edge and Elo expectation.

Captures the single strongest signal in the problem — the rating difference —
plus the standard Elo expected score. Inputs are the **reconstructed pre-game
ratings** (``white_pregame_rating`` / ``black_pregame_rating`` from
:mod:`features.pregame`), *not* the game object's post-game ``white_rating`` /
``black_rating``. Building on the pre-game values keeps this family leakage-safe:
there is no post-game rating in the feature set to difference against.
"""

from __future__ import annotations

import pandas as pd


def elo_expected_score(white_rating: pd.Series, black_rating: pd.Series) -> pd.Series:
    """White's Elo expected score against black (0..1)."""
    return 1.0 / (1.0 + 10.0 ** ((black_rating - white_rating) / 400.0))


def add_rating_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rating-difference and Elo-expectation features from pre-game ratings."""
    result = df.copy()
    white = result["white_pregame_rating"]
    black = result["black_pregame_rating"]
    rating_diff = white - black
    result["rating_diff"] = rating_diff
    result["abs_rating_diff"] = rating_diff.abs()
    result["avg_rating"] = (white + black) / 2.0
    result["white_expected_score"] = elo_expected_score(white, black)
    result["black_expected_score"] = 1.0 - result["white_expected_score"]
    return result
