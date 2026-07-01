from pathlib import Path
from types import TracebackType
from urllib.request import Request

import pytest

import chess_api
from chess_api import (
    cached_fetch_json,
    cached_fetch_optional_json,
    fetch_json,
    fetch_json_with_retries,
    iter_group_urls,
    iter_round_urls,
    read_json_object,
)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_fetch_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        assert request.headers["User-agent"] == "test-agent"
        assert timeout == 30
        return FakeResponse(b'{"rounds": ["round-url"]}')

    monkeypatch.setattr(chess_api, "urlopen", fake_urlopen)

    assert fetch_json("https://example.test", user_agent="test-agent") == {
        "rounds": ["round-url"]
    }


def test_fetch_json_rejects_non_object(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        return FakeResponse(b'["not", "an", "object"]')

    monkeypatch.setattr(chess_api, "urlopen", fake_urlopen)

    with pytest.raises(TypeError, match="Expected a JSON object"):
        fetch_json("https://example.test")


def test_cached_fetch_json_uses_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "payload.json"
    cache_path.write_text('{"cached": true}\n', encoding="utf-8")

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        raise AssertionError("cache hit should not call urlopen")

    monkeypatch.setattr(chess_api, "urlopen", fake_urlopen)

    assert cached_fetch_json("https://example.test", cache_path) == {"cached": True}


def test_cached_fetch_json_refreshes_and_writes_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "payload.json"

    def fake_urlopen(request: Request, timeout: int) -> FakeResponse:
        assert request.headers["User-agent"] == "test-agent"
        return FakeResponse(b'{"cached": false}')

    monkeypatch.setattr(chess_api, "urlopen", fake_urlopen)

    assert cached_fetch_json(
        "https://example.test",
        cache_path,
        refresh=True,
        user_agent="test-agent",
    ) == {"cached": False}
    assert read_json_object(cache_path) == {"cached": False}


def test_read_json_object_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(TypeError, match="Expected JSON object"):
        read_json_object(path)


def test_fetch_json_with_retries_recovers_after_transient_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = {"count": 0}

    def flaky_fetch_json(url: str, user_agent: str) -> dict[str, object]:
        attempts["count"] += 1
        if attempts["count"] < 2:
            raise RuntimeError("transient")
        return {"ok": True}

    monkeypatch.setattr(chess_api, "fetch_json", flaky_fetch_json)
    monkeypatch.setattr("chess_api.time.sleep", lambda _seconds: None)

    assert fetch_json_with_retries("https://example.test", retries=3) == {"ok": True}
    assert attempts["count"] == 2


def test_fetch_json_with_retries_raises_after_exhausting_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fail(url: str, user_agent: str) -> dict[str, object]:
        raise RuntimeError("down")

    monkeypatch.setattr(chess_api, "fetch_json", always_fail)
    monkeypatch.setattr("chess_api.time.sleep", lambda _seconds: None)

    with pytest.raises(RuntimeError, match="down"):
        fetch_json_with_retries("https://example.test", retries=2)


def test_cached_fetch_optional_json_returns_none_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def always_fail(url: str, user_agent: str) -> dict[str, object]:
        raise RuntimeError("missing")

    monkeypatch.setattr(chess_api, "fetch_json", always_fail)
    monkeypatch.setattr("chess_api.time.sleep", lambda _seconds: None)

    assert (
        cached_fetch_optional_json(
            "https://example.test",
            tmp_path / "missing.json",
        )
        is None
    )


def test_iter_endpoint_urls() -> None:
    assert iter_round_urls({"rounds": ["round-a", "round-b"]}) == [
        "round-a",
        "round-b",
    ]
    assert iter_round_urls({}) == []
    assert iter_round_urls({"rounds": "not-a-list"}) == []
    assert iter_group_urls({"groups": ["group-a", "group-b"]}) == [
        "group-a",
        "group-b",
    ]
    assert iter_group_urls({}) == []
    assert iter_group_urls({"groups": "not-a-list"}) == []
