"""Tests for the diff-only cross-event recent-form family."""

from __future__ import annotations

import pandas as pd
import pytest

from features.recent_form import add_recent_form_features

CORE_DIFFS = [
    "recent_last5_score_rate_diff",
    "recent_last5_draw_rate_diff",
    "recent_last5_avg_opponent_rating_diff",
    "current_color_last5_score_rate_diff",
    "prior_event_score_rate_diff",
]


def _game(
    *,
    event: str,
    start_time: int,
    rnd: int,
    white: str,
    black: str,
    white_rating: int,
    black_rating: int,
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
        "white_rating": white_rating,
        "black_rating": black_rating,
        "target": target,
    }


def _scenario() -> pd.DataFrame:
    # alice: feb r1 win vs bob, feb r2 loss vs carol, mar r1 win vs dave.
    return pd.DataFrame(
        [
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="alice",
                black="bob",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="f1",
            ),
            _game(
                event="feb",
                start_time=100,
                rnd=2,
                white="alice",
                black="carol",
                white_rating=2510,
                black_rating=2450,
                target="white_loss",
                game_url="f2",
            ),
            _game(
                event="mar",
                start_time=200,
                rnd=1,
                white="alice",
                black="dave",
                white_rating=2520,
                black_rating=2300,
                target="white_win",
                game_url="m1",
            ),
        ]
    )


def test_emits_only_diff_core_no_raw_columns() -> None:
    out = add_recent_form_features(_scenario())

    for column in CORE_DIFFS:
        assert column in out.columns
    assert not [
        c for c in out.columns if "recent_last5" in c and not c.endswith("_diff")
    ]
    assert not [
        c
        for c in out.columns
        if c.startswith(("white_prior_event", "black_prior_event"))
    ]
    # The dropped count signal is gone entirely.
    assert "prior_event_games_played_diff" not in out.columns
    assert "white_prior_event_games_played" not in out.columns


def test_first_ever_game_is_cold_start() -> None:
    out = add_recent_form_features(_scenario())
    row = out[out["game_url"] == "f1"].iloc[0]

    for column in CORE_DIFFS:
        assert row[column] == 0.0


def test_recent_form_uses_only_strictly_earlier_games() -> None:
    out = add_recent_form_features(_scenario())
    f2 = out[out["game_url"] == "f2"].iloc[0]

    # white=alice (one prior feb r1 win -> rate 1.0), black=carol (no history).
    assert f2["recent_last5_score_rate_diff"] == pytest.approx(1.0)
    # feb r1 is the same event, so no prior-event signal yet for alice.
    assert f2["prior_event_score_rate_diff"] == 0.0


def test_recent_form_spans_events() -> None:
    out = add_recent_form_features(_scenario())
    m1 = out[out["game_url"] == "m1"].iloc[0]

    # alice's prior games are the two February games (win, loss); dave has none.
    assert m1["recent_last5_score_rate_diff"] == pytest.approx(0.5)
    assert m1["recent_last5_avg_opponent_rating_diff"] == pytest.approx(
        (2400 + 2450) / 2
    )
    assert m1["current_color_last5_score_rate_diff"] == pytest.approx(0.5)
    assert m1["prior_event_score_rate_diff"] == pytest.approx(0.5)


def test_no_leakage_when_later_games_change() -> None:
    full = add_recent_form_features(_scenario())
    f1_full = full[full["game_url"] == "f1"].iloc[0]

    trimmed = add_recent_form_features(_scenario()[lambda d: d["game_url"] == "f1"])
    f1_trimmed = trimmed[trimmed["game_url"] == "f1"].iloc[0]

    for col in CORE_DIFFS:
        assert f1_full[col] == f1_trimmed[col]
