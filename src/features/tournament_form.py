"""In-tournament prior-form family: a player's shape *before* the current round.

For each game we summarize how each player has done so far **within the same
tournament**, using only their games in strictly-earlier rounds. All games in
round N are featurized from state through round N-1, so the current round never
leaks into its own features (see :mod:`features._temporal`).

This family is deliberately trimmed to a small, robust core, emitted as
white-minus-black **diffs** only (the raw per-color values were highly redundant
with each other and with the diff). The kept signals are the ones prior work and
rating theory support: points so far, games played, strength of schedule, and
momentum.

**No rating-level prior features.** Chess.com game objects report each player's
rating *after* the game (verified: the sign of a player's round-over-round rating
change matches that round's result >99% of the time). A prior round's rating is
therefore effectively the current game's *pre-game* rating, so any feature that
lets the model difference it against the current (post-game) rating reconstructs
the result. Earlier versions carried ``prior_last_or_pregame_rating`` and
``prior_rolling3_rating``; both were removed because they leaked the label this
way. Genuine strength is already captured leak-safely by the current-game ratings
in :mod:`features.rating`, so nothing legitimate is lost.

Cold-start: at round 1 a player has no prior games, so every signal here is a
neutral 0.
"""

from __future__ import annotations

import pandas as pd

from features._temporal import merge_color_features, player_game_log

# Core per-player signals; each is emitted only as a white-minus-black diff.
# Deliberately excludes any rating-level signal (see module docstring: prior
# ratings leak the result once differenced against the current post-game rating).
_CORE_FEATURES = [
    "prior_score",
    "prior_games_played",
    "prior_avg_opponent_rating",
    "prior_current_streak",
]


def add_tournament_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the diff-only in-tournament prior-form core."""
    log = player_game_log(df)
    per_entry = _prior_state(log)
    merged = merge_color_features(df, per_entry, _CORE_FEATURES)
    for feature in _CORE_FEATURES:
        merged[f"{feature}_diff"] = (
            merged[f"white_{feature}"] - merged[f"black_{feature}"]
        )
    raw_columns = [
        f"{prefix}{feature}"
        for feature in _CORE_FEATURES
        for prefix in ("white_", "black_")
    ]
    return merged.drop(columns=raw_columns)


def _prior_state(log: pd.DataFrame) -> pd.DataFrame:
    """Per (game, player) prior-in-tournament core state from earlier rounds only."""
    states: list[dict[str, object]] = []
    for _keys, group in log.groupby(["event", "player"], sort=False):
        ordered = group.sort_values("round")
        scores = ordered["score"].tolist()
        opponents = ordered["opponent_rating"].tolist()
        game_urls = ordered["game_url"].tolist()
        colors = ordered["color"].tolist()

        for i in range(len(ordered)):
            prior_scores = scores[:i]
            prior_opponents = opponents[:i]

            states.append(
                {
                    "game_url": game_urls[i],
                    "color": colors[i],
                    "prior_games_played": i,
                    "prior_score": sum(prior_scores),
                    "prior_avg_opponent_rating": _mean(prior_opponents),
                    "prior_current_streak": _streak(prior_scores),
                }
            )
    return pd.DataFrame(states)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _streak(scores: list[float]) -> float:
    """Signed trailing run: +k for k straight wins, -k for k straight losses.

    A draw at the tail yields 0 (no active decisive streak).
    """
    if not scores:
        return 0.0
    last = scores[-1]
    if last == 0.5:
        return 0.0
    length = 0
    for score in reversed(scores):
        if score == last:
            length += 1
        else:
            break
    return float(length) if last == 1.0 else float(-length)
