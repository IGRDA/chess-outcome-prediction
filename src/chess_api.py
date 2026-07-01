"""Helpers for fetching and caching Chess.com PubAPI data."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

JsonObject = dict[str, Any]

DEFAULT_USER_AGENT = "chess-outcome-prediction/0.1"


def fetch_json(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_seconds: int = 30,
) -> JsonObject:
    """Fetch a JSON object from the Chess.com public API."""
    request = Request(
        url,
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = response.read()
    data: object = json.loads(payload)
    if not isinstance(data, dict):
        msg = f"Expected a JSON object from {url}"
        raise TypeError(msg)
    return cast(JsonObject, data)


def fetch_json_with_retries(
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    retries: int = 3,
    retry_sleep_seconds: float = 0.5,
) -> JsonObject:
    """Fetch JSON, retrying transient failures a small number of times."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            return fetch_json(url=url, user_agent=user_agent)
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(retry_sleep_seconds * (attempt + 1))
    if last_error is None:
        msg = f"Could not fetch {url}"
        raise RuntimeError(msg)
    raise last_error


def read_json_object(path: Path) -> JsonObject:
    """Read a JSON object from disk."""
    data: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Expected JSON object in {path}"
        raise TypeError(msg)
    return cast(JsonObject, data)


def write_json_object(path: Path, payload: JsonObject) -> None:
    """Write a JSON object to disk with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def cached_fetch_json(
    url: str,
    cache_path: Path,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> JsonObject:
    """Fetch JSON through a filesystem cache."""
    if cache_path.exists() and not refresh:
        return read_json_object(cache_path)

    payload = fetch_json_with_retries(url=url, user_agent=user_agent)
    write_json_object(cache_path, payload)
    return payload


def cached_fetch_optional_json(
    url: str,
    cache_path: Path,
    refresh: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
) -> JsonObject | None:
    """Fetch optional enrichment JSON, returning None after retries fail."""
    try:
        return cached_fetch_json(
            url=url,
            cache_path=cache_path,
            refresh=refresh,
            user_agent=user_agent,
        )
    except Exception:
        return None


def iter_round_urls(tournament: JsonObject) -> list[str]:
    """Return round URLs from tournament metadata."""
    rounds = tournament.get("rounds", [])
    if not isinstance(rounds, list):
        return []
    return [str(url) for url in rounds]


def iter_group_urls(round_payload: JsonObject) -> list[str]:
    """Return group URLs from a round payload."""
    groups = round_payload.get("groups", [])
    if not isinstance(groups, list):
        return []
    return [str(url) for url in groups]
