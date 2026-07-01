import csv
import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

import make_dataset
from features import flatten_game


def test_collect_data_traverses_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        make_dataset,
        "EVENTS",
        (("sample_event", "https://api.test/tournament/sample"),),
    )

    def fake_cached_fetch_json(
        url: str,
        cache_path: Path,
        refresh: bool,
        user_agent: str,
    ) -> dict[str, Any]:
        assert refresh is True
        assert user_agent == "test-agent"
        if url == "https://api.test/tournament/sample":
            assert cache_path.name == "sample_event.json"
            return {
                "url": "https://www.chess.com/tournament/sample",
                "start_time": 100_000,
                "rounds": ["https://api.test/tournament/sample/1"],
            }
        if url == "https://api.test/tournament/sample/1":
            assert cache_path.name == "sample_event_round_01.json"
            return {"groups": ["https://api.test/tournament/sample/1/1"]}
        if url == "https://api.test/tournament/sample/1/1":
            assert cache_path.name == "sample_event_round_01_group_01.json"
            return {
                "games": [sample_game("Alice", "Bob", 1500, 1400, "win", "resigned")]
            }
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(make_dataset, "cached_fetch_json", fake_cached_fetch_json)
    monkeypatch.setattr(
        make_dataset,
        "collect_player_enrichment",
        lambda data_dir, usernames, refresh, user_agent: (  # noqa: ARG005
            {username: None for username in usernames},
            {username: None for username in usernames},
        ),
    )

    records, profiles, stats = make_dataset.collect_data(
        data_dir=tmp_path,
        refresh=True,
        user_agent="test-agent",
    )

    assert len(records) == 1
    assert records[0].event == "sample_event"
    assert records[0].round == 1
    assert records[0].group == 1
    assert records[0].target == "white_win"
    assert profiles == {"alice": None, "bob": None}
    assert stats == {"alice": None, "bob": None}


def test_collect_player_enrichment_fetches_profiles_and_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_optional_fetch(
        url: str,
        cache_path: Path,
        refresh: bool,
        user_agent: str,
    ) -> dict[str, Any]:
        assert refresh is False
        assert user_agent == "test-agent"
        calls.append((url, cache_path))
        if url.endswith("/stats"):
            return {"chess_blitz": {"record": {"win": 1, "loss": 0, "draw": 0}}}
        return {"username": "alice"}

    monkeypatch.setattr(
        make_dataset,
        "cached_fetch_optional_json",
        fake_optional_fetch,
    )

    profiles, stats = make_dataset.collect_player_enrichment(
        data_dir=tmp_path,
        usernames=["alice"],
        refresh=False,
        user_agent="test-agent",
    )

    assert profiles["alice"] == {"username": "alice"}
    assert stats["alice"] == {
        "chess_blitz": {"record": {"win": 1, "loss": 0, "draw": 0}}
    }
    assert [path.parent.name for _, path in calls] == ["players", "player_stats"]


def test_main_writes_processed_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://www.chess.com/tournament/sample",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=sample_game("Alice", "Bob", 1500, 1400, "win", "resigned"),
    )

    monkeypatch.setattr(
        make_dataset,
        "parse_args",
        lambda: Namespace(refresh=False, data_dir=tmp_path, user_agent="test-agent"),
    )
    monkeypatch.setattr(
        make_dataset,
        "collect_data",
        lambda data_dir, refresh, user_agent: (  # noqa: ARG005
            [record],
            {"alice": {"title": "GM"}, "bob": None},
            {"alice": None, "bob": None},
        ),
    )

    make_dataset.main()

    processed = tmp_path / "processed"
    assert (processed / "games.csv").exists()
    assert (processed / "run_summary.json").exists()
    assert "Built datasets: 1 games, 2 players" in capsys.readouterr().out
    summary = json.loads((processed / "run_summary.json").read_text(encoding="utf-8"))
    assert not any(Path(path).is_absolute() for path in summary["outputs"].values())

    with (processed / "modeling_dataset.csv").open(encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert header == make_dataset.MODELING_COLUMNS


def test_build_run_summary_uses_repo_relative_output_paths() -> None:
    summary = make_dataset.build_run_summary(
        records=[],
        profiles={},
        stats={},
        processed_dir=make_dataset.REPO_ROOT / "data" / "processed",
    )

    assert summary["outputs"] == {
        "games": "data/processed/games.csv",
        "modeling_dataset": "data/processed/modeling_dataset.csv",
        "run_summary": "data/processed/run_summary.json",
    }


def test_small_helpers(tmp_path: Path) -> None:
    assert make_dataset.parse_url_tail_int("https://api.test/path/11") == 11
    assert make_dataset.safe_cache_name("Alice/../Bob!") == "alice_.._bob"

    output = tmp_path / "nested" / "payload.json"
    make_dataset.write_json(output, {"ok": True})
    assert output.read_text(encoding="utf-8") == '{\n  "ok": true\n}\n'


def sample_game(
    white_username: str,
    black_username: str,
    white_rating: int,
    black_rating: int,
    white_result: str,
    black_result: str,
) -> dict[str, object]:
    result = "1-0" if white_result == "win" else "0-1"
    return {
        "url": f"https://example.test/{white_username}-vs-{black_username}",
        "pgn": (
            '[Event "Live Chess"]\n'
            f'[White "{white_username}"]\n'
            f'[Black "{black_username}"]\n'
            f'[Result "{result}"]\n\n'
            f"1. e4 {result}\n"
        ),
        "end_time": 123_456,
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "white": {
            "rating": white_rating,
            "result": white_result,
            "username": white_username,
            "uuid": f"{white_username}-uuid",
        },
        "black": {
            "rating": black_rating,
            "result": black_result,
            "username": black_username,
            "uuid": f"{black_username}-uuid",
        },
        "eco": "https://www.chess.com/openings/sample",
    }
