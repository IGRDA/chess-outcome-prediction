"""Head-to-head family: how White has fared against *this* opponent before.

For each game we look at prior meetings between the same two players, ordered by
``(event_order, round)`` and restricted to strictly-earlier games. Scores are
expressed from the current game's White perspective, so a reverse-color prior
meeting is flipped accordingly.

With only two tournaments this signal is very sparse (few pairs meet twice), so
it is kept as a candidate for the later modeling stage rather than asserted
useful. Cold-start (no prior meetings) yields neutral 0 rates and a 0 count to
gate on.
"""

from __future__ import annotations

import pandas as pd

from features._temporal import WHITE_SCORE, add_event_order

_FEATURES = [
    "h2h_games",
    "h2h_white_score_rate",
    "h2h_same_color_games",
    "h2h_same_color_white_score_rate",
]


def add_head_to_head_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add prior head-to-head features (White perspective) with cold-start 0s."""
    work = pd.DataFrame(
        {
            "game_url": df["game_url"].to_numpy(),
            "event_order": add_event_order(df).to_numpy(),
            "round": df["round"].to_numpy(),
            "white": df["white_username"].to_numpy(),
            "black": df["black_username"].to_numpy(),
            "white_score": df["target"].map(WHITE_SCORE).to_numpy(),
        }
    )
    work["pair"] = [
        tuple(sorted(pair)) for pair in zip(work["white"], work["black"], strict=True)
    ]

    rows: list[dict[str, object]] = []
    for _pair, group in work.groupby("pair", sort=False):
        meetings = group.sort_values(["event_order", "round"]).to_dict("records")
        for i, current in enumerate(meetings):
            current_white = current["white"]
            all_scores: list[float] = []
            same_color_scores: list[float] = []
            for prior in meetings[:i]:
                if prior["white"] == current_white:
                    score = prior["white_score"]
                    same_color_scores.append(score)
                else:
                    score = 1.0 - prior["white_score"]
                all_scores.append(score)
            rows.append(
                {
                    "game_url": current["game_url"],
                    "h2h_games": len(all_scores),
                    "h2h_white_score_rate": _mean(all_scores),
                    "h2h_same_color_games": len(same_color_scores),
                    "h2h_same_color_white_score_rate": _mean(same_color_scores),
                }
            )

    features = pd.DataFrame(rows)
    return df.merge(features, on="game_url", how="left")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
