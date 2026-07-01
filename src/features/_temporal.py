"""Shared temporal helpers for the leakage-safe feature layer.

The backbone is the *player-game log*: one row per ``(game, player)`` describing
that game from that player's perspective (their rating, the opponent's rating,
and the score they earned). Form-style feature families aggregate this log over
games that finished **strictly before** the current game, which is what keeps
them leakage-safe.

Ordering key: ``(event_order, round)``.

- Events are ordered by ``tournament_start_time`` so an earlier tournament always
  precedes a later one (no reliance on event-name spelling).
- Within an event, an earlier round precedes a later one.
- Games that share the same ``(event_order, round)`` are treated as
  *concurrent* and never see one another: every player plays at most one game
  per round and those games start at the same instant, so a feature for round N
  may only use rounds ``< N``.
"""

from __future__ import annotations

import pandas as pd

# White's score for each label; black's score is the complement.
WHITE_SCORE: dict[str, float] = {"white_win": 1.0, "draw": 0.5, "white_loss": 0.0}


def add_event_order(df: pd.DataFrame) -> pd.Series:
    """Return an integer chronological rank of each row's event.

    Events are ranked by their earliest ``tournament_start_time`` so the result
    is 0 for the first tournament, 1 for the next, and so on.
    """
    ordered_events = (
        df.groupby("event")["tournament_start_time"].min().sort_values().index
    )
    ranks = {event: rank for rank, event in enumerate(ordered_events)}
    return df["event"].map(ranks).astype("int64")


def player_game_log(df: pd.DataFrame) -> pd.DataFrame:
    """Explode one-row-per-game into two rows-per-game, one per player.

    Columns: ``game_url, event, event_order, round, player, color,
    player_rating, opponent, opponent_rating, score, win, loss, draw``.
    """
    event_order = add_event_order(df)
    white_score = df["target"].map(WHITE_SCORE)

    white = pd.DataFrame(
        {
            "game_url": df["game_url"].to_numpy(),
            "event": df["event"].to_numpy(),
            "event_order": event_order.to_numpy(),
            "round": df["round"].to_numpy(),
            "player": df["white_username"].to_numpy(),
            "color": "white",
            "player_rating": df["white_rating"].to_numpy(),
            "opponent": df["black_username"].to_numpy(),
            "opponent_rating": df["black_rating"].to_numpy(),
            "score": white_score.to_numpy(),
        }
    )
    black = pd.DataFrame(
        {
            "game_url": df["game_url"].to_numpy(),
            "event": df["event"].to_numpy(),
            "event_order": event_order.to_numpy(),
            "round": df["round"].to_numpy(),
            "player": df["black_username"].to_numpy(),
            "color": "black",
            "player_rating": df["black_rating"].to_numpy(),
            "opponent": df["white_username"].to_numpy(),
            "opponent_rating": df["white_rating"].to_numpy(),
            "score": (1.0 - white_score).to_numpy(),
        }
    )

    log = pd.concat([white, black], ignore_index=True)
    log["win"] = (log["score"] == 1.0).astype("int64")
    log["loss"] = (log["score"] == 0.0).astype("int64")
    log["draw"] = (log["score"] == 0.5).astype("int64")
    return log


def merge_color_features(
    df: pd.DataFrame,
    per_entry: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Attach per-(game, color) features back onto the wide game frame.

    ``per_entry`` must have ``game_url``, ``color`` and the given
    ``feature_columns``. White entries land under a ``white_`` prefix, black
    under ``black_``. Returns ``df`` with the prefixed columns added.
    """
    result = df.copy()
    for color, prefix in (("white", "white_"), ("black", "black_")):
        side = per_entry.loc[
            per_entry["color"] == color, ["game_url", *feature_columns]
        ]
        side = side.rename(columns={c: f"{prefix}{c}" for c in feature_columns})
        result = result.merge(side, on="game_url", how="left")
    return result
