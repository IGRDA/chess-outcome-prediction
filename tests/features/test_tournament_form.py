"""Tests for the diff-only in-tournament prior-form core, including no-leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from features.tournament_form import add_tournament_form_features

CORE_DIFFS = [
    "prior_last_or_pregame_rating_diff",
    "prior_rolling3_rating_diff",
    "prior_score_diff",
    "prior_games_played_diff",
    "prior_avg_opponent_rating_diff",
    "prior_current_streak_diff",
]


def _game(
    *,
    rnd: int,
    white: str,
    black: str,
    white_rating: int,
    black_rating: int,
    target: str,
    game_url: str,
    event: str = "feb",
    start_time: int = 100,
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
    # r1: alice (white) beats bob. r2 g2: carol (white) loses to alice (black).
    # r2 g3: bob (white) beats dave -- concurrent with g2, must not affect it.
    return pd.DataFrame(
        [
            _game(
                rnd=1,
                white="alice",
                black="bob",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="g1",
            ),
            _game(
                rnd=2,
                white="carol",
                black="alice",
                white_rating=2450,
                black_rating=2480,
                target="white_loss",
                game_url="g2",
            ),
            _game(
                rnd=2,
                white="bob",
                black="dave",
                white_rating=2380,
                black_rating=2300,
                target="white_win",
                game_url="g3",
            ),
        ]
    )


def test_emits_only_diff_core_no_raw_columns() -> None:
    out = add_tournament_form_features(_scenario())

    for column in CORE_DIFFS:
        assert column in out.columns
    # No raw per-color prior columns should survive.
    assert not [c for c in out.columns if c.startswith(("white_prior", "black_prior"))]
    # Dropped signals must be absent entirely.
    for dropped in (
        "prior_win_rate_diff",
        "prior_draw_rate_diff",
        "prior_last_score_diff",
        "color_balance_diff",
    ):
        assert dropped not in out.columns


def test_round1_cold_start_uses_pregame_rating_fallback() -> None:
    out = add_tournament_form_features(_scenario())
    row = out[out["game_url"] == "g1"].iloc[0]

    assert row["prior_games_played_diff"] == 0
    assert row["prior_score_diff"] == 0
    assert row["prior_rolling3_rating_diff"] == 0
    # Fallback rating diff equals the pre-game rating diff at round 1.
    assert row["prior_last_or_pregame_rating_diff"] == 2500 - 2400


def test_diffs_reflect_only_earlier_rounds() -> None:
    out = add_tournament_form_features(_scenario())
    row = out[out["game_url"] == "g2"].iloc[0]

    # white=carol (cold start), black=alice (one prior round-1 win as white).
    assert row["prior_games_played_diff"] == 0 - 1
    assert row["prior_score_diff"] == pytest.approx(0.0 - 1.0)
    assert row["prior_avg_opponent_rating_diff"] == pytest.approx(0.0 - 2400)
    assert row["prior_current_streak_diff"] == pytest.approx(0.0 - 1.0)
    assert row["prior_rolling3_rating_diff"] == pytest.approx(0.0 - 2500)
    # carol pre-game 2450 vs alice last-observed 2500.
    assert row["prior_last_or_pregame_rating_diff"] == pytest.approx(2450 - 2500)


def test_rolling_and_streak_over_multiple_rounds() -> None:
    df = pd.DataFrame(
        [
            _game(
                rnd=1,
                white="alice",
                black="o1",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="a1",
            ),
            _game(
                rnd=2,
                white="alice",
                black="o2",
                white_rating=2510,
                black_rating=2400,
                target="white_loss",
                game_url="a2",
            ),
            _game(
                rnd=3,
                white="alice",
                black="o3",
                white_rating=2520,
                black_rating=2400,
                target="white_win",
                game_url="a3",
            ),
            _game(
                rnd=4,
                white="alice",
                black="o4",
                white_rating=2530,
                black_rating=2400,
                target="white_win",
                game_url="a4",
            ),
        ]
    )

    out = add_tournament_form_features(df)
    row = out[out["game_url"] == "a4"].iloc[0]  # white=alice (3 priors), black=o4 (0)

    assert row["prior_games_played_diff"] == 3
    assert row["prior_score_diff"] == pytest.approx(2.0)  # win, loss, win
    assert row["prior_rolling3_rating_diff"] == pytest.approx((2500 + 2510 + 2520) / 3)
    assert row["prior_current_streak_diff"] == pytest.approx(1.0)  # last win only
    assert row["prior_avg_opponent_rating_diff"] == pytest.approx(2400)


def test_draw_at_tail_breaks_the_streak() -> None:
    df = pd.DataFrame(
        [
            _game(
                rnd=1,
                white="alice",
                black="o1",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="a1",
            ),
            _game(
                rnd=2,
                white="alice",
                black="o2",
                white_rating=2500,
                black_rating=2400,
                target="draw",
                game_url="a2",
            ),
            _game(
                rnd=3,
                white="alice",
                black="o3",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="a3",
            ),
        ]
    )

    out = add_tournament_form_features(df)
    a3 = out[out["game_url"] == "a3"].iloc[0]  # alice: win then draw -> streak 0

    assert a3["prior_current_streak_diff"] == 0.0


def test_no_leakage_when_later_rounds_change() -> None:
    full = add_tournament_form_features(_scenario())
    round1_full = full[full["game_url"] == "g1"].iloc[0]

    only_r1 = add_tournament_form_features(_scenario()[lambda d: d["round"] == 1])
    round1_trimmed = only_r1[only_r1["game_url"] == "g1"].iloc[0]
    for col in CORE_DIFFS:
        assert round1_full[col] == round1_trimmed[col]

    altered = _scenario()
    altered.loc[altered["game_url"] == "g2", "target"] = "white_win"
    round1_altered = add_tournament_form_features(altered)
    round1_altered_row = round1_altered[round1_altered["game_url"] == "g1"].iloc[0]
    for col in CORE_DIFFS:
        assert round1_full[col] == round1_altered_row[col]


def test_concurrent_same_round_game_does_not_leak() -> None:
    out = add_tournament_form_features(_scenario())
    g2 = out[out["game_url"] == "g2"].iloc[0]

    # carol (white in g2) has no prior games; bob's concurrent g3 must not leak in.
    # white_prior_games_played would be 0 -> diff equals -(alice's 1 prior).
    assert g2["prior_games_played_diff"] == -1
