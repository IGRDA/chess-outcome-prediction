"""Tests for the rating / base feature family."""

from __future__ import annotations

import pandas as pd
import pytest

from features.rating import add_rating_features


def test_rating_features_basic_values() -> None:
    df = pd.DataFrame({"white_pregame_rating": [3000], "black_pregame_rating": [2800]})

    out = add_rating_features(df).iloc[0]

    assert out["rating_diff"] == 200
    assert out["abs_rating_diff"] == 200
    assert out["avg_rating"] == 2900
    assert out["white_expected_score"] == pytest.approx(0.7597, abs=0.0001)
    assert out["black_expected_score"] == pytest.approx(1 - 0.7597, abs=0.0001)


def test_rating_features_equal_ratings_are_even() -> None:
    df = pd.DataFrame({"white_pregame_rating": [2500], "black_pregame_rating": [2500]})

    out = add_rating_features(df).iloc[0]

    assert out["rating_diff"] == 0
    assert out["abs_rating_diff"] == 0
    assert out["white_expected_score"] == pytest.approx(0.5)


def test_rating_features_negative_diff_for_underdog_white() -> None:
    df = pd.DataFrame({"white_pregame_rating": [2600], "black_pregame_rating": [2900]})

    out = add_rating_features(df).iloc[0]

    assert out["rating_diff"] == -300
    assert out["abs_rating_diff"] == 300
    assert out["white_expected_score"] < 0.5
