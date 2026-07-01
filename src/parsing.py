"""Flatten raw Chess.com game payloads into typed records and CSV rows.

This module is a pure *data collection* concern: it parses game JSON into a
``GameRecord`` and projects records onto CSV rows (an audit view and a
leak-safe base view). Engineered features live in the separate ``features``
package so collection and feature engineering stay orthogonal.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

Outcome = Literal["white_win", "white_loss", "draw"]
CsvValue: TypeAlias = str | int | float | bool | None
JsonObject: TypeAlias = dict[str, Any]
CsvRow: TypeAlias = dict[str, CsvValue]

DRAW_RESULT_CODES = {
    "50move",
    "agreed",
    "insufficient",
    "repetition",
    "stalemate",
    "timevsinsufficient",
}


@dataclass(frozen=True)
class GameRecord:
    """Flattened representation of one game."""

    event: str
    tournament_url: str
    tournament_start_time: int
    round: int
    group: int
    game_url: str
    white_username: str
    black_username: str
    white_uuid: str | None
    black_uuid: str | None
    white_rating: int
    black_rating: int
    target: Outcome
    white_result_code: str
    black_result_code: str
    pgn: str | None
    fen: str | None
    eco: str | None
    end_time: int | None
    pgn_result: str | None
    termination: str | None

    def to_games_row(self) -> CsvRow:
        """Return the audit-friendly processed game row."""
        return {
            "event": self.event,
            "tournament_url": self.tournament_url,
            "tournament_start_time": self.tournament_start_time,
            "round": self.round,
            "group": self.group,
            "game_url": self.game_url,
            "white_username": self.white_username,
            "black_username": self.black_username,
            "white_uuid": self.white_uuid,
            "black_uuid": self.black_uuid,
            "white_rating": self.white_rating,
            "black_rating": self.black_rating,
            "target": self.target,
            "white_result_code": self.white_result_code,
            "black_result_code": self.black_result_code,
            "fen": self.fen,
            "eco": self.eco,
            "end_time": self.end_time,
            "pgn_result": self.pgn_result,
            "termination": self.termination,
        }

    def to_base_row(self) -> CsvRow:
        """Return the minimal, leak-safe base row (identifiers + label)."""
        return {
            "event": self.event,
            "tournament_url": self.tournament_url,
            "tournament_start_time": self.tournament_start_time,
            "round": self.round,
            "group": self.group,
            "game_url": self.game_url,
            "white_username": self.white_username,
            "black_username": self.black_username,
            "white_uuid": self.white_uuid,
            "black_uuid": self.black_uuid,
            "white_rating": self.white_rating,
            "black_rating": self.black_rating,
            "target": self.target,
        }


def normalize_username(username: str) -> str:
    """Normalize Chess.com usernames for joining and cache paths."""
    return username.strip().lower()


def outcome_from_result(result: str) -> Outcome:
    """Map PGN result notation to a white-perspective outcome label."""
    result_map: dict[str, Outcome] = {
        "1-0": "white_win",
        "0-1": "white_loss",
        "1/2-1/2": "draw",
    }
    try:
        return result_map[result]
    except KeyError as exc:
        msg = f"Unsupported result: {result}"
        raise ValueError(msg) from exc


def outcome_from_result_codes(white_result: str, black_result: str) -> Outcome:
    """Map Chess.com player result codes to a white-perspective label."""
    if white_result == "win":
        return "white_win"
    if black_result == "win":
        return "white_loss"
    if white_result in DRAW_RESULT_CODES and black_result in DRAW_RESULT_CODES:
        return "draw"

    msg = f"Unsupported result codes: white={white_result}, black={black_result}"
    raise ValueError(msg)


def flatten_game(
    event: str,
    tournament_url: str,
    tournament_start_time: int,
    round_number: int,
    group_number: int,
    game: Mapping[str, Any],
) -> GameRecord:
    """Flatten a game payload into a typed game record."""
    white = _mapping(game.get("white"))
    black = _mapping(game.get("black"))
    pgn = _optional_str(game.get("pgn"))
    pgn_result = extract_pgn_tag(pgn, "Result")

    try:
        target = outcome_from_result_codes(
            _required_str(white, "result"),
            _required_str(black, "result"),
        )
    except ValueError:
        if pgn_result is None:
            raise
        target = outcome_from_result(pgn_result)

    return GameRecord(
        event=event,
        tournament_url=tournament_url,
        tournament_start_time=tournament_start_time,
        round=round_number,
        group=group_number,
        game_url=_required_str(game, "url"),
        white_username=normalize_username(_required_str(white, "username")),
        black_username=normalize_username(_required_str(black, "username")),
        white_uuid=_optional_str(white.get("uuid")),
        black_uuid=_optional_str(black.get("uuid")),
        white_rating=_required_int(white, "rating"),
        black_rating=_required_int(black, "rating"),
        target=target,
        white_result_code=_required_str(white, "result"),
        black_result_code=_required_str(black, "result"),
        pgn=pgn,
        fen=_optional_str(game.get("fen")),
        eco=_optional_str(game.get("eco")),
        end_time=_optional_int(game.get("end_time")),
        pgn_result=pgn_result,
        termination=extract_pgn_tag(pgn, "Termination"),
    )


def extract_pgn_tag(pgn: str | None, tag: str) -> str | None:
    """Extract a PGN header tag value."""
    if not pgn:
        return None
    match = re.search(rf'^\[{re.escape(tag)} "([^"]+)"\]', pgn, flags=re.MULTILINE)
    return match.group(1) if match else None


def build_games_rows(records: Sequence[GameRecord]) -> list[CsvRow]:
    """Build audit-friendly game rows."""
    return [record.to_games_row() for record in records]


def build_base_rows(records: Sequence[GameRecord]) -> list[CsvRow]:
    """Build minimal, leak-safe base rows (one row per game)."""
    return [record.to_base_row() for record in records]


def _mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value)
    return {}


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if value is None:
        msg = f"Missing required string field: {key}"
        raise ValueError(msg)
    return str(value)


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = _optional_int(payload.get(key))
    if value is None:
        msg = f"Missing required integer field: {key}"
        raise ValueError(msg)
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        return int(value)
    return None
