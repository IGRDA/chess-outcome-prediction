"""Build local raw caches and processed datasets for the take-home exercise."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from chess_api import (
    DEFAULT_USER_AGENT,
    JsonObject,
    cached_fetch_json,
    cached_fetch_optional_json,
    iter_group_urls,
    iter_round_urls,
)
from features import (
    CsvRow,
    GameRecord,
    build_games_rows,
    build_modeling_rows,
    flatten_game,
    normalize_username,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

EVENTS = (
    (
        "feb_2026",
        "https://api.chess.com/pub/tournament/"
        "titled-tuesday-blitz-february-10-2026-6221327",
    ),
    (
        "mar_2026",
        "https://api.chess.com/pub/tournament/"
        "titled-tuesday-blitz-march-10-2026-6277141",
    ),
)

GAMES_COLUMNS = [
    "event",
    "tournament_url",
    "tournament_start_time",
    "round",
    "group",
    "game_url",
    "white_username",
    "black_username",
    "white_uuid",
    "black_uuid",
    "white_rating",
    "black_rating",
    "target",
    "white_result_code",
    "black_result_code",
    "fen",
    "eco",
    "end_time",
    "pgn_result",
    "termination",
]

MODELING_COLUMNS = [
    "event",
    "tournament_url",
    "round",
    "group",
    "game_url",
    "white_username",
    "black_username",
    "white_uuid",
    "black_uuid",
    "white_rating",
    "black_rating",
    "target",
]


@dataclass(frozen=True)
class EventConfig:
    """Tournament configuration."""

    name: str
    url: str


def main() -> None:
    """Run the local data pipeline."""
    args = parse_args()
    data_dir = args.data_dir

    records, profiles, stats = collect_data(
        data_dir=data_dir,
        refresh=args.refresh,
        user_agent=args.user_agent,
    )

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    games_rows = build_games_rows(records)
    modeling_rows = build_modeling_rows(records)
    summary = build_run_summary(records, profiles, stats, processed_dir)

    write_csv(processed_dir / "games.csv", games_rows, GAMES_COLUMNS)
    write_csv(processed_dir / "modeling_dataset.csv", modeling_rows, MODELING_COLUMNS)
    write_json(processed_dir / "run_summary.json", summary)

    print(
        "Built datasets: "
        f"{len(records)} games, "
        f"{len(unique_usernames(records))} players, "
        f"{summary['missing_profiles']} missing profiles, "
        f"{summary['missing_player_stats']} missing stat payloads"
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh cached API responses.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory for raw caches and processed datasets.",
    )
    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header for Chess.com PubAPI requests.",
    )
    return parser.parse_args()


def collect_data(
    data_dir: Path,
    refresh: bool,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[
    list[GameRecord], dict[str, JsonObject | None], dict[str, JsonObject | None]
]:
    """Collect raw endpoint data and flatten game records."""
    raw_dir = data_dir / "raw"
    records: list[GameRecord] = []

    for event in [EventConfig(name=name, url=url) for name, url in EVENTS]:
        tournament = cached_fetch_json(
            url=event.url,
            cache_path=raw_dir / "tournaments" / f"{event.name}.json",
            refresh=refresh,
            user_agent=user_agent,
        )
        tournament_url = str(tournament.get("url", event.url))
        tournament_start_time = _int_or_zero(tournament.get("start_time"))

        for round_url in iter_round_urls(tournament):
            round_number = parse_url_tail_int(round_url)
            round_payload = cached_fetch_json(
                url=round_url,
                cache_path=raw_dir
                / "rounds"
                / f"{event.name}_round_{round_number:02d}.json",
                refresh=refresh,
                user_agent=user_agent,
            )

            for group_url in iter_group_urls(round_payload):
                group_number = parse_url_tail_int(group_url)
                group_cache_name = (
                    f"{event.name}_round_{round_number:02d}"
                    f"_group_{group_number:02d}.json"
                )
                group_payload = cached_fetch_json(
                    url=group_url,
                    cache_path=raw_dir / "groups" / group_cache_name,
                    refresh=refresh,
                    user_agent=user_agent,
                )
                games = group_payload.get("games", [])
                if not isinstance(games, list):
                    continue
                for game in games:
                    if isinstance(game, dict):
                        records.append(
                            flatten_game(
                                event=event.name,
                                tournament_url=tournament_url,
                                tournament_start_time=tournament_start_time,
                                round_number=round_number,
                                group_number=group_number,
                                game=game,
                            )
                        )

    profiles, stats = collect_player_enrichment(
        data_dir=data_dir,
        usernames=unique_usernames(records),
        refresh=refresh,
        user_agent=user_agent,
    )
    return records, profiles, stats


def collect_player_enrichment(
    data_dir: Path,
    usernames: list[str],
    refresh: bool,
    user_agent: str = DEFAULT_USER_AGENT,
) -> tuple[dict[str, JsonObject | None], dict[str, JsonObject | None]]:
    """Collect optional player profile and stats payloads."""
    raw_dir = data_dir / "raw"
    profiles: dict[str, JsonObject | None] = {}
    stats: dict[str, JsonObject | None] = {}

    for username in usernames:
        quoted_username = quote(username, safe="")
        cache_name = safe_cache_name(username)
        profiles[username] = cached_fetch_optional_json(
            url=f"https://api.chess.com/pub/player/{quoted_username}",
            cache_path=raw_dir / "players" / f"{cache_name}.json",
            refresh=refresh,
            user_agent=user_agent,
        )
        stats[username] = cached_fetch_optional_json(
            url=f"https://api.chess.com/pub/player/{quoted_username}/stats",
            cache_path=raw_dir / "player_stats" / f"{cache_name}.json",
            refresh=refresh,
            user_agent=user_agent,
        )

    return profiles, stats


def build_run_summary(
    records: list[GameRecord],
    profiles: dict[str, JsonObject | None],
    stats: dict[str, JsonObject | None],
    processed_dir: Path,
) -> dict[str, Any]:
    """Build a compact run summary."""
    events: dict[str, dict[str, Any]] = {}
    for event in sorted({record.event for record in records}):
        event_records = [record for record in records if record.event == event]
        event_players = {
            username
            for record in event_records
            for username in (record.white_username, record.black_username)
        }
        target_counts = Counter(record.target for record in event_records)
        events[event] = {
            "games": len(event_records),
            "rounds": len({record.round for record in event_records}),
            "players": len(event_players),
            "class_balance": dict(sorted(target_counts.items())),
        }

    return {
        "events": events,
        "total_games": len(records),
        "unique_players": len(unique_usernames(records)),
        "missing_profiles": sum(profile is None for profile in profiles.values()),
        "missing_player_stats": sum(payload is None for payload in stats.values()),
        "outputs": {
            "games": relative_output_path(processed_dir / "games.csv"),
            "modeling_dataset": relative_output_path(
                processed_dir / "modeling_dataset.csv"
            ),
            "run_summary": relative_output_path(processed_dir / "run_summary.json"),
        },
    }


def relative_output_path(path: Path, base_dir: Path = REPO_ROOT) -> str:
    """Return a stable relative path for generated summaries."""
    return Path(os.path.relpath(path.resolve(), start=base_dir.resolve())).as_posix()


def write_csv(path: Path, rows: list[CsvRow], fieldnames: list[str]) -> None:
    """Write rows to CSV with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON object to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def unique_usernames(records: list[GameRecord]) -> list[str]:
    """Return sorted unique player usernames from records."""
    return sorted(
        {
            normalize_username(username)
            for record in records
            for username in (record.white_username, record.black_username)
        }
    )


def parse_url_tail_int(url: str) -> int:
    """Parse the final URL path segment as an integer."""
    tail = url.rstrip("/").rsplit("/", maxsplit=1)[-1]
    return int(tail)


def safe_cache_name(value: str) -> str:
    """Return a filesystem-safe cache name."""
    return re.sub(r"[^a-z0-9_.-]+", "_", value.lower()).strip("_")


def _int_or_zero(value: object) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return 0


if __name__ == "__main__":
    main()
