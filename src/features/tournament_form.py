"""In-tournament prior-form family: a player's shape *before* the current round.

For each game we summarize how each player has done so far **within the same
tournament**, using only their games in strictly-earlier rounds. All games in
round N are featurized from state through round N-1, so the current round never
leaks into its own features (see :mod:`features._temporal`).

This family is deliberately trimmed to a small, robust core, emitted as
white-minus-black **diffs** only (the raw per-color values were highly redundant
with each other and with the diff). The kept signals are the ones prior work and
rating theory support: recent in-event rating (last and rolling-3), points so
far, games played, strength of schedule, and momentum.

Cold-start: at round 1 a player has no prior games. ``prior_last_or_pregame_rating``
falls back to the game's **own pre-game rating** (a leakage-safe, always-present
value) so the rating signal is never a misleading 0; the other prior signals are
a neutral 0.
"""

from __future__ import annotations

import pandas as pd

from features._temporal import merge_color_features, player_game_log

# Core per-player signals; each is emitted only as a white-minus-black diff.
_CORE_FEATURES = [
    "prior_last_or_pregame_rating",
    "prior_rolling3_rating",
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
        ratings = ordered["player_rating"].tolist()
        opponents = ordered["opponent_rating"].tolist()
        game_urls = ordered["game_url"].tolist()
        colors = ordered["color"].tolist()

        for i in range(len(ordered)):
            prior_scores = scores[:i]
            prior_ratings = ratings[:i]
            prior_opponents = opponents[:i]
            has_prior = i > 0

            states.append(
                {
                    "game_url": game_urls[i],
                    "color": colors[i],
                    "prior_games_played": i,
                    "prior_score": sum(prior_scores),
                    "prior_rolling3_rating": _mean(prior_ratings[-3:]),
                    "prior_avg_opponent_rating": _mean(prior_opponents),
                    "prior_current_streak": _streak(prior_scores),
                    # Cold-start: fall back to the game's own pre-game rating.
                    "prior_last_or_pregame_rating": (
                        prior_ratings[-1] if has_prior else ratings[i]
                    ),
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
