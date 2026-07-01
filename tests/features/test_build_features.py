"""End-to-end tests for the assembled feature matrix, split tagging, and CLI."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

import build_features
from features import build_feature_matrix
from features.build import LEAKY_COLUMNS


def _row(
    *, event: str, start: int, white: str, black: str, target: str, url: str
) -> dict[str, object]:
    return {
        "event": event,
        "tournament_start_time": start,
        "round": 1,
        "game_url": url,
        "white_username": white,
        "black_username": black,
        "white_rating": 2500,
        "black_rating": 2400,
        "target": target,
    }


def _base_frame() -> pd.DataFrame:
    # alice plays across a history event, a train event, and the test event.
    return pd.DataFrame(
        [
            _row(
                event="tt_2026_02_03",
                start=100,
                white="alice",
                black="bob",
                target="white_win",
                url="h1",
            ),
            _row(
                event="tt_2026_02_10",
                start=200,
                white="alice",
                black="carol",
                target="white_win",
                url="t1",
            ),
            _row(
                event="tt_2026_03_10",
                start=300,
                white="alice",
                black="dave",
                target="white_win",
                url="x1",
            ),
        ]
    )


def test_build_feature_matrix_adds_families_without_leakage() -> None:
    out = build_feature_matrix(_base_frame())

    # build_feature_matrix keeps every event (filtering happens in the CLI).
    assert len(out) == 3
    for column in (
        "rating_diff",
        "white_expected_score",
        "rating_edge_scaled_by_round",
        "prior_last_or_pregame_rating_diff",
        "recent_last5_score_rate_diff",
        "h2h_games",
    ):
        assert column in out.columns
    assert LEAKY_COLUMNS.isdisjoint(out.columns)


def test_build_feature_matrix_rejects_leaky_input() -> None:
    leaky = _base_frame()
    leaky["pgn"] = "1. e4 e5"

    with pytest.raises(ValueError, match="leaky columns"):
        build_feature_matrix(leaky)


def test_cli_emits_only_labeled_rows_with_split_and_history_features(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True)
    _base_frame().to_csv(processed / "base_dataset.csv", index=False)

    monkeypatch.setattr(
        build_features, "parse_args", lambda: Namespace(data_dir=tmp_path)
    )

    build_features.main()

    written = pd.read_csv(processed / "modeling_dataset.csv")

    # History-only event contributes features but emits no rows.
    assert set(written["event"]) == {"tt_2026_02_10", "tt_2026_03_10"}
    assert set(written["split"]) == {"train", "test"}
    assert "h1" not in set(written["game_url"])

    # The train row's cross-event features reflect the history event.
    train_row = written[written["event"] == "tt_2026_02_10"].iloc[0]
    assert train_row["prior_event_score_rate_diff"] == pytest.approx(1.0)
    assert train_row["recent_last5_score_rate_diff"] == pytest.approx(1.0)

    assert "Built modeling dataset: 2 rows" in capsys.readouterr().out
