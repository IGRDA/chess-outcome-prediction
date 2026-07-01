"""Build local raw caches and processed datasets for the take-home exercise."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
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
from events import DISCOVERY_SEED_PLAYERS, EVENTS, SPLIT_BY_EVENT, Event
from parsing import (
    CsvRow,
    GameRecord,
    build_base_rows,
    build_games_rows,
    flatten_game,
    normalize_username,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"

# Regex to pull the date fragment (zero-padded day) out of a TT tournament slug,
# e.g. ".../titled-tuesday-blitz-february-03-2026-6221001" -> "february-03-2026".
TT_SLUG_RE = re.compile(r"titled-tuesday-blitz-([a-z]+-\d{2}-\d{4})-\d+$")

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

BASE_COLUMNS = [
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
]


def main() -> None:
    """Run the local data pipeline."""
    args = parse_args()
    data_dir = args.data_dir

    records = collect_data(
        data_dir=data_dir,
        refresh=args.refresh,
        user_agent=args.user_agent,
    )

    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    games_rows = build_games_rows(records)
    base_rows = build_base_rows(records)
    summary = build_run_summary(records, processed_dir)

    write_csv(processed_dir / "games.csv", games_rows, GAMES_COLUMNS)
    write_csv(processed_dir / "base_dataset.csv", base_rows, BASE_COLUMNS)
    write_json(processed_dir / "run_summary.json", summary)

    print(
        "Built datasets: "
        f"{len(records)} games, "
        f"{len(unique_usernames(records))} players"
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
) -> list[GameRecord]:
    """Collect raw endpoint data and flatten game records for all events."""
    raw_dir = data_dir / "raw"
    records: list[GameRecord] = []

    event_urls = discover_event_urls(EVENTS, raw_dir, refresh, user_agent)

    for event in EVENTS:
        event_url = event_urls[event.name]
        tournament = cached_fetch_json(
            url=event_url,
            cache_path=raw_dir / "tournaments" / f"{event.name}.json",
            refresh=refresh,
            user_agent=user_agent,
        )
        tournament_url = str(tournament.get("url", event_url))
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

    return records


def discover_event_urls(
    events: tuple[Event, ...],
    raw_dir: Path,
    refresh: bool,
    user_agent: str = DEFAULT_USER_AGENT,
) -> dict[str, str]:
    """Resolve each event to its TT tournament API URL via player tournament lists.

    Titled Tuesday tournament ids are not derivable from the date, so we union
    the ``finished`` lists of several TT regulars and match each event's
    zero-padded date slug. Using multiple seeds covers weeks any one player
    skipped. Raises if a target event cannot be resolved.
    """
    urls_by_date: dict[str, str] = {}
    for player in DISCOVERY_SEED_PLAYERS:
        tournaments_url = (
            f"https://api.chess.com/pub/player/{quote(player, safe='')}/tournaments"
        )
        payload = cached_fetch_optional_json(
            url=tournaments_url,
            cache_path=raw_dir
            / "player_tournaments"
            / f"{safe_cache_name(player)}.json",
            refresh=refresh,
            user_agent=user_agent,
        )
        if payload is None:
            continue
        for slug_fragment, url in titled_tuesday_urls_by_date(payload).items():
            urls_by_date.setdefault(slug_fragment, url)

    resolved: dict[str, str] = {}
    missing: list[str] = []
    for event in events:
        event_url = urls_by_date.get(event.slug_fragment)
        if event_url is None:
            missing.append(event.slug_fragment)
        else:
            resolved[event.name] = event_url
    if missing:
        msg = f"Could not discover Titled Tuesday tournaments for: {missing}"
        raise RuntimeError(msg)
    return resolved


def titled_tuesday_urls_by_date(tournaments_payload: JsonObject) -> dict[str, str]:
    """Map TT date slug fragment -> tournament API URL from a /tournaments payload."""
    result: dict[str, str] = {}
    finished = tournaments_payload.get("finished", [])
    if not isinstance(finished, list):
        return result
    for entry in finished:
        if not isinstance(entry, dict):
            continue
        api_url = entry.get("@id")
        if not isinstance(api_url, str):
            continue
        match = TT_SLUG_RE.search(api_url)
        if match:
            result.setdefault(match.group(1), api_url)
    return result


def build_run_summary(
    records: list[GameRecord],
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
            "role": SPLIT_BY_EVENT.get(event, "unknown"),
            "games": len(event_records),
            "rounds": len({record.round for record in event_records}),
            "players": len(event_players),
            "class_balance": dict(sorted(target_counts.items())),
        }

    return {
        "events": events,
        "total_games": len(records),
        "unique_players": len(unique_usernames(records)),
        "outputs": {
            "games": relative_output_path(processed_dir / "games.csv"),
            "base_dataset": relative_output_path(processed_dir / "base_dataset.csv"),
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
