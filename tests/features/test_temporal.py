"""Tests for the shared player-game log backbone."""

from __future__ import annotations

import pandas as pd

from features._temporal import add_event_order, merge_color_features, player_game_log


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


def test_add_event_order_ranks_by_start_time() -> None:
    df = pd.DataFrame(
        [
            _game(
                event="mar",
                start_time=200,
                rnd=1,
                white="a",
                black="b",
                white_rating=2000,
                black_rating=1900,
                target="white_win",
                game_url="g_mar",
            ),
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="a",
                black="b",
                white_rating=2000,
                black_rating=1900,
                target="white_win",
                game_url="g_feb",
            ),
        ]
    )

    order = add_event_order(df)

    # Earlier start_time -> lower rank, regardless of row position or name.
    assert order.tolist() == [1, 0]


def test_player_game_log_explodes_both_perspectives() -> None:
    df = pd.DataFrame(
        [
            _game(
                event="feb",
                start_time=100,
                rnd=3,
                white="alice",
                black="bob",
                white_rating=2500,
                black_rating=2400,
                target="white_win",
                game_url="g1",
            )
        ]
    )

    log = player_game_log(df)

    assert len(log) == 2
    white = log[log["color"] == "white"].iloc[0]
    black = log[log["color"] == "black"].iloc[0]

    assert white["player"] == "alice"
    assert white["opponent"] == "bob"
    assert white["player_rating"] == 2500
    assert white["opponent_rating"] == 2400
    assert white["score"] == 1.0
    assert white["win"] == 1 and white["loss"] == 0 and white["draw"] == 0
    assert white["round"] == 3 and white["event_order"] == 0

    assert black["player"] == "bob"
    assert black["score"] == 0.0
    assert black["loss"] == 1 and black["win"] == 0


def test_player_game_log_draw_splits_score() -> None:
    df = pd.DataFrame(
        [
            _game(
                event="feb",
                start_time=100,
                rnd=1,
                white="alice",
                black="bob",
                white_rating=2500,
                black_rating=2400,
                target="draw",
                game_url="g1",
            )
        ]
    )

    log = player_game_log(df)

    assert set(log["score"]) == {0.5}
    assert set(log["draw"]) == {1}
    assert set(log["win"]) == {0} and set(log["loss"]) == {0}


def test_merge_color_features_prefixes_by_color() -> None:
    df = pd.DataFrame({"game_url": ["g1"], "other": [7]})
    per_entry = pd.DataFrame(
        {
            "game_url": ["g1", "g1"],
            "color": ["white", "black"],
            "form": [1.5, -2.0],
        }
    )

    merged = merge_color_features(df, per_entry, ["form"])

    assert merged.loc[0, "white_form"] == 1.5
    assert merged.loc[0, "black_form"] == -2.0
    assert merged.loc[0, "other"] == 7
