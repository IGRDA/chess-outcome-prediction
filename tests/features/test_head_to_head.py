"""Tests for the head-to-head family."""

from __future__ import annotations

import pandas as pd
import pytest

from features.head_to_head import add_head_to_head_features


def _game(
    *,
    event: str,
    start_time: int,
    rnd: int,
    white: str,
    black: str,
    target: str,
    game_url: str,
) -> dict[str, object]:
    return {
        "event": event,
        "tournament_start_time": start_time,
        "round": rnd,
        "game_url": game_url,
        "white_username": white,
        "black_username": black,
        "white_rating": 2500,
        "black_rating": 2400,
        "target": target,
    }


def test_no_prior_meeting_is_cold_start() -> None:
    df = pd.DataFrame(
        [
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="alice",
                black="bob",
                target="white_win",
                game_url="g1",
            ),
        ]
    )

    out = add_head_to_head_features(df).iloc[0]

    assert out["h2h_games"] == 0
    assert out["h2h_white_score_rate"] == 0.0
    assert out["h2h_same_color_games"] == 0


def test_reverse_color_prior_meeting_is_flipped() -> None:
    df = pd.DataFrame(
        [
            # feb: alice (white) beats bob.
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="alice",
                black="bob",
                target="white_win",
                game_url="g1",
            ),
            # mar: bob (white) vs alice -- prior meeting had bob as black (a loss).
            _game(
                event="mar",
                start_time=200,
                rnd=1,
                white="bob",
                black="alice",
                target="white_win",
                game_url="g2",
            ),
        ]
    )

    out = add_head_to_head_features(df)
    g2 = out[out["game_url"] == "g2"].iloc[0]

    assert g2["h2h_games"] == 1
    # From current white (bob)'s view, the prior meeting was a loss -> 0.0.
    assert g2["h2h_white_score_rate"] == pytest.approx(0.0)
    # Colors differ (bob was black then, white now), so no same-color history.
    assert g2["h2h_same_color_games"] == 0


def test_same_color_history_accumulates() -> None:
    df = pd.DataFrame(
        [
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="alice",
                black="bob",
                target="white_win",
                game_url="g1",
            ),
            _game(
                event="feb",
                start_time=100,
                rnd=2,
                white="alice",
                black="bob",
                target="white_loss",
                game_url="g2",
            ),
            _game(
                event="mar",
                start_time=200,
                rnd=1,
                white="alice",
                black="bob",
                target="white_win",
                game_url="g3",
            ),
        ]
    )

    out = add_head_to_head_features(df)
    g3 = out[out["game_url"] == "g3"].iloc[0]

    # Two prior meetings, both with alice as white (win then loss).
    assert g3["h2h_games"] == 2
    assert g3["h2h_white_score_rate"] == pytest.approx(0.5)
    assert g3["h2h_same_color_games"] == 2
    assert g3["h2h_same_color_white_score_rate"] == pytest.approx(0.5)
