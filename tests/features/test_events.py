"""Tests for the event-series config."""

from __future__ import annotations

from datetime import date

from events import EVENTS, SPLIT_BY_EVENT, TARGET_ROLES, Event


def test_slug_fragment_zero_pads_day() -> None:
    assert Event("x", date(2026, 2, 3), "history").slug_fragment == "february-03-2026"
    assert Event("x", date(2026, 2, 10), "train").slug_fragment == "february-10-2026"
    assert Event("x", date(2026, 3, 3), "val").slug_fragment == "march-03-2026"


def test_event_series_roles() -> None:
    roles = [event.role for event in EVENTS]
    # 3 history, 3 train, 1 val, 1 test, in chronological order.
    assert roles == [
        "history",
        "history",
        "history",
        "train",
        "train",
        "train",
        "val",
        "test",
    ]
    # Names sort chronologically (date-based), matching event order.
    assert list(SPLIT_BY_EVENT) == [event.name for event in EVENTS]
    assert sorted(TARGET_ROLES) == ["test", "train", "val"]
    # Events are strictly increasing in date.
    days = [event.day for event in EVENTS]
    assert days == sorted(days)
