"""Tests for leak-safe pre-game rating reconstruction."""

from __future__ import annotations

import pandas as pd

from features.pregame import add_pregame_rating_columns


def _game(
    *,
    event: str,
    start: int,
    rnd: int,
    white: str,
    black: str,
    white_rating: int,
    black_rating: int,
    game_url: str,
    target: str = "white_win",
) -> dict[str, object]:
    return {
        "event": event,
        "tournament_start_time": start,
        "round": rnd,
        "game_url": game_url,
        "white_username": white,
        "black_username": black,
        "white_rating": white_rating,
        "black_rating": black_rating,
        "target": target,
    }


def _scenario() -> pd.DataFrame:
    # alice: e1 r1 (post 2500) -> e1 r2 (post 2510) -> e2 r1 (post 2505, as black).
    return pd.DataFrame(
        [
            _game(
                event="e1",
                start=100,
                rnd=1,
                white="alice",
                black="bob",
                white_rating=2500,
                black_rating=2400,
                game_url="g1",
            ),
            _game(
                event="e1",
                start=100,
                rnd=2,
                white="alice",
                black="carol",
                white_rating=2510,
                black_rating=2450,
                game_url="g2",
            ),
            _game(
                event="e2",
                start=200,
                rnd=1,
                white="dave",
                black="alice",
                white_rating=2600,
                black_rating=2505,
                game_url="g3",
            ),
        ]
    )


def _row(out: pd.DataFrame, url: str) -> pd.Series:
    return out[out["game_url"] == url].iloc[0]


def test_debut_falls_back_to_own_rating() -> None:
    out = add_pregame_rating_columns(_scenario())
    g1 = _row(out, "g1")
    # First appearance for both players -> pre-game rating is the game's own.
    assert g1["white_pregame_rating"] == 2500
    assert g1["black_pregame_rating"] == 2400


def test_within_event_uses_prior_round_rating() -> None:
    out = add_pregame_rating_columns(_scenario())
    # alice at e1 r2: pre-game rating is her post-game rating from e1 r1 (2500).
    assert _row(out, "g2")["white_pregame_rating"] == 2500


def test_cross_event_prior_week_feeds_round_one() -> None:
    out = add_pregame_rating_columns(_scenario())
    # alice (black) at e2 r1: most recent earlier game is e1 r2 (post 2510).
    assert _row(out, "g3")["black_pregame_rating"] == 2510
    # dave debuts at e2 r1 -> own rating.
    assert _row(out, "g3")["white_pregame_rating"] == 2600


def test_pregame_rating_ignores_the_games_own_rating() -> None:
    # Leakage guard: a game's own (post-game) rating must not affect its own
    # pre-game rating, which comes only from strictly-earlier games.
    base = _scenario()
    altered = base.copy()
    altered.loc[altered["game_url"] == "g2", "white_rating"] = 9999

    unchanged = _row(add_pregame_rating_columns(base), "g2")["white_pregame_rating"]
    still = _row(add_pregame_rating_columns(altered), "g2")["white_pregame_rating"]
    assert unchanged == still == 2500
