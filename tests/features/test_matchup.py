"""Tests for the underdog / matchup-shape feature family."""

from __future__ import annotations

import pandas as pd
import pytest

from features.matchup import add_matchup_features


def _frame() -> pd.DataFrame:
    # One event with 4 rounds; white favored in the round-1 row.
    return pd.DataFrame(
        {
            "event": ["feb", "feb", "feb"],
            "round": [1, 3, 4],
            "white_pregame_rating": [2600, 2500, 2400],
            "black_pregame_rating": [2400, 2600, 2400],
        }
    )


def test_matchup_favorite_and_magnitude() -> None:
    out = add_matchup_features(_frame())

    assert out.loc[0, "white_is_favorite"] == 1
    assert out.loc[0, "favorite_magnitude"] == 200
    assert out.loc[1, "white_is_favorite"] == 0  # white lower-rated
    assert out.loc[2, "white_is_favorite"] == 0  # equal -> not strictly favored


def test_matchup_stage_scaling_uses_event_max_round() -> None:
    out = add_matchup_features(_frame())

    # max round in the event is 4.
    assert out.loc[0, "round_norm"] == pytest.approx(0.25)
    assert out.loc[2, "round_norm"] == pytest.approx(1.0)
    assert out.loc[0, "rounds_remaining"] == 3
    assert out.loc[2, "rounds_remaining"] == 0
    # rating_edge_scaled_by_round = rating_diff * round_norm
    assert out.loc[0, "rating_edge_scaled_by_round"] == pytest.approx(200 * 0.25)
    assert out.loc[0, "is_late_round"] == 0
    assert out.loc[2, "is_late_round"] == 1


def test_matchup_round_norm_is_per_event() -> None:
    df = pd.DataFrame(
        {
            "event": ["feb", "mar"],
            "round": [2, 2],
            "white_pregame_rating": [2500, 2500],
            "black_pregame_rating": [2400, 2400],
        }
    )

    out = add_matchup_features(df)

    # Each event has a single round here, so its own max is that round.
    assert out.loc[0, "round_norm"] == pytest.approx(1.0)
    assert out.loc[1, "round_norm"] == pytest.approx(1.0)
