"""Assemble the feature families into the modeling-ready feature matrix.

:func:`build_feature_matrix` applies each family in order, highest-signal first.
It refuses to run on a frame that still carries post-game (leaky) columns, so
the feature layer can only ever be built from the leak-safe base dataset.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from features.head_to_head import add_head_to_head_features
from features.matchup import add_matchup_features
from features.rating import add_rating_features
from features.recent_form import add_recent_form_features
from features.tournament_form import add_tournament_form_features

FeatureBuilder = Callable[[pd.DataFrame], pd.DataFrame]

FEATURE_BUILDERS: list[FeatureBuilder] = [
    add_rating_features,
    add_matchup_features,
    add_tournament_form_features,
    add_recent_form_features,
    add_head_to_head_features,
]

# Post-game fields that must never reach the feature layer.
LEAKY_COLUMNS = frozenset(
    {
        "pgn",
        "fen",
        "eco",
        "end_time",
        "termination",
        "white_result_code",
        "black_result_code",
        "pgn_result",
    }
)


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every feature family to the leak-safe base dataset."""
    leaked = LEAKY_COLUMNS.intersection(df.columns)
    if leaked:
        raise ValueError(f"Base dataset contains leaky columns: {sorted(leaked)}")

    result = df
    for builder in FEATURE_BUILDERS:
        result = builder(result)
    return result
