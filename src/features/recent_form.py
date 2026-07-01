"""Recent-form family: a player's shape across *all* prior Titled Tuesday games.

Like :mod:`features.tournament_form` but the window spans every event, not just
the current one. Titled Tuesday events are comparable by construction, so a
player's earlier-week results are legitimate prior signal for a later game.
History is ordered by ``(event_order, round)`` and only strictly-earlier games
are used, so this stays leakage-safe.

Emitted as white-minus-black **diffs** only (the raw per-color values were
redundant with each other and the diff). Cold-start (no prior games) yields a
neutral 0 for every signal, so the diff is simply 0 when neither player has
history.
"""

from __future__ import annotations

import pandas as pd

from features._temporal import merge_color_features, player_game_log

# Core per-player signals; each is emitted only as a white-minus-black diff.
_CORE_FEATURES = [
    "recent_last5_score_rate",
    "recent_last5_draw_rate",
    "recent_last5_avg_opponent_rating",
    "current_color_last5_score_rate",
    "prior_event_score_rate",
]


def add_recent_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the diff-only cross-event recent-form core."""
    log = player_game_log(df)
    per_entry = _recent_state(log)
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


def _recent_state(log: pd.DataFrame) -> pd.DataFrame:
    states: list[dict[str, object]] = []
    for _player, group in log.groupby("player", sort=False):
        ordered = group.sort_values(["event_order", "round"])
        scores = ordered["score"].tolist()
        draws = ordered["draw"].tolist()
        opponents = ordered["opponent_rating"].tolist()
        colors = ordered["color"].tolist()
        event_orders = ordered["event_order"].tolist()
        game_urls = ordered["game_url"].tolist()

        for i in range(len(ordered)):
            current_color = colors[i]
            current_event = event_orders[i]
            same_color_scores = [
                scores[j] for j in range(i) if colors[j] == current_color
            ]
            prior_event_scores = [
                scores[j] for j in range(i) if event_orders[j] < current_event
            ]
            states.append(
                {
                    "game_url": game_urls[i],
                    "color": current_color,
                    "recent_last5_score_rate": _mean(scores[:i][-5:]),
                    "recent_last5_draw_rate": _mean(draws[:i][-5:]),
                    "recent_last5_avg_opponent_rating": _mean(opponents[:i][-5:]),
                    "current_color_last5_score_rate": _mean(same_color_scores[-5:]),
                    "prior_event_score_rate": _mean(prior_event_scores),
                }
            )
    return pd.DataFrame(states)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
