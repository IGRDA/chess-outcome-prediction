"""The Titled Tuesday event series used for features and modeling.

Titled Tuesday is a weekly 11-round Swiss. We use eight consecutive weekly
events. The earliest three are **history-only**: they power the prior-event /
recent-form / head-to-head features but produce no modeling rows. The rest are
the labeled rows, split temporally into train / validation / test.

Keeping this as the single source of truth (imported by both collection and the
feature build) is what lets the two stages stay orthogonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

_MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


@dataclass(frozen=True)
class Event:
    """One Titled Tuesday event and its role in the modeling dataset."""

    name: str
    day: date
    role: str  # "history" | "train" | "val" | "test"

    @property
    def slug_fragment(self) -> str:
        """The date portion of a TT tournament slug, e.g. ``february-03-2026``.

        Days are zero-padded to match Chess.com's slug format. Built by hand
        (not ``strftime``) to stay locale-independent and deterministic.
        """
        return f"{_MONTHS[self.day.month - 1]}-{self.day.day:02d}-{self.day.year}"


EVENTS: tuple[Event, ...] = (
    Event("tt_2026_01_20", date(2026, 1, 20), "history"),
    Event("tt_2026_01_27", date(2026, 1, 27), "history"),
    Event("tt_2026_02_03", date(2026, 2, 3), "history"),
    Event("tt_2026_02_10", date(2026, 2, 10), "train"),
    Event("tt_2026_02_17", date(2026, 2, 17), "train"),
    Event("tt_2026_02_24", date(2026, 2, 24), "train"),
    Event("tt_2026_03_03", date(2026, 3, 3), "val"),
    Event("tt_2026_03_10", date(2026, 3, 10), "test"),
)

# Roles that produce modeling rows (history-only events are features-only).
TARGET_ROLES: frozenset[str] = frozenset({"train", "val", "test"})

# event name -> split label, for tagging modeling rows.
SPLIT_BY_EVENT: dict[str, str] = {event.name: event.role for event in EVENTS}

# TT regulars whose /tournaments lists are unioned to discover event URLs.
# Several seeds cover any single week a given player skipped.
DISCOVERY_SEED_PLAYERS: tuple[str, ...] = (
    "hikaru",
    "gm_dmitrij",
    "chesswarrior7197",
    "fandorine",
    "oleksandr_bortnyk",
    "lachesisq",
)
