"""Pre-game rating family: reconstruct each player's true pre-game rating.

Chess.com game objects report each player's rating **after** the game, so the
current game's ``white_rating`` / ``black_rating`` are post-game values and carry
a residue of *this* game's result. A player's rating from their most recent
strictly-earlier game — ordered by ``(event_order, round)`` across all events, so
round 1 of a later event is fed by the previous week — is exactly their pre-game
rating for this game, and is leakage-safe.

This family runs **first** so the rating and matchup families build on the
leak-safe pre-game ratings rather than the post-game values. The raw post-game
columns are kept for reference but must be excluded from model features (exposing
both a pre- and post-game rating lets a model difference them to recover the
result — see :mod:`features.tournament_form`).

Cold-start: a player's first appearance anywhere in the collected window has no
earlier game, so we fall back to the game's own rating — the best available
strength estimate, and non-exploitable once the post-game columns are dropped
from the feature set.
"""

from __future__ import annotations

import pandas as pd

from features._temporal import merge_color_features, player_game_log


def add_pregame_rating_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``white_pregame_rating`` / ``black_pregame_rating`` (leak-safe)."""
    log = player_game_log(df).sort_values(["player", "event_order", "round"])
    prior = log.groupby("player")["player_rating"].shift(1)
    # Cold-start (no earlier game): fall back to the game's own rating.
    log["pregame_rating"] = prior.fillna(log["player_rating"])
    return merge_color_features(df, log, ["pregame_rating"])
