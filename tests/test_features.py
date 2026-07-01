import pytest

from features import (
    build_games_rows,
    build_modeling_rows,
    extract_pgn_tag,
    flatten_game,
    normalize_username,
    outcome_from_result,
    outcome_from_result_codes,
)

MODELING_COLUMNS = {
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
}

LEAKY_FIELDS = {
    "pgn",
    "fen",
    "eco",
    "end_time",
    "termination",
    "white_result_code",
    "black_result_code",
    "pgn_result",
}


def test_outcome_from_result() -> None:
    assert outcome_from_result("1-0") == "white_win"
    assert outcome_from_result("0-1") == "white_loss"
    assert outcome_from_result("1/2-1/2") == "draw"


def test_outcome_from_result_rejects_unknown_result() -> None:
    with pytest.raises(ValueError, match="Unsupported result"):
        outcome_from_result("*")


@pytest.mark.parametrize(
    ("white_result", "black_result", "expected"),
    [
        ("win", "resigned", "white_win"),
        ("resigned", "win", "white_loss"),
        ("win", "timeout", "white_win"),
        ("timeout", "win", "white_loss"),
        ("win", "checkmated", "white_win"),
        ("checkmated", "win", "white_loss"),
        ("repetition", "repetition", "draw"),
        ("insufficient", "insufficient", "draw"),
        ("timevsinsufficient", "timevsinsufficient", "draw"),
    ],
)
def test_outcome_from_result_codes(
    white_result: str,
    black_result: str,
    expected: str,
) -> None:
    assert outcome_from_result_codes(white_result, black_result) == expected


def test_outcome_from_result_codes_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError, match="Unsupported result codes"):
        outcome_from_result_codes("resigned", "resigned")


def test_normalize_username_trims_and_lowercases() -> None:
    assert normalize_username("  MagnusCarlsen  ") == "magnuscarlsen"


def test_extract_pgn_tag() -> None:
    pgn = '[Event "Live Chess"]\n[Result "1/2-1/2"]\n\n1. e4 1/2-1/2\n'
    assert extract_pgn_tag(pgn, "Result") == "1/2-1/2"
    assert extract_pgn_tag(pgn, "Termination") is None
    assert extract_pgn_tag(None, "Result") is None


def test_flatten_game_handles_draw() -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=sample_game("Alice", "Bob", 1500, 1400, "agreed", "agreed"),
    )

    assert record.target == "draw"


def test_flatten_game_falls_back_to_pgn_result() -> None:
    game = sample_game("Alice", "Bob", 1500, 1400, "win", "resigned")
    game["white"] = {**game["white"], "result": "mystery"}  # type: ignore[dict-item]
    game["black"] = {**game["black"], "result": "mystery"}  # type: ignore[dict-item]

    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=game,
    )

    assert record.target == "white_win"


def test_flatten_game_requires_rating() -> None:
    game = sample_game("Alice", "Bob", 1500, 1400, "win", "resigned")
    game["white"] = {"result": "win", "username": "Alice", "uuid": "alice-uuid"}

    with pytest.raises(ValueError, match="Missing required integer field: rating"):
        flatten_game(
            event="sample",
            tournament_url="https://example.test/tournament",
            tournament_start_time=100_000,
            round_number=1,
            group_number=1,
            game=game,
        )


def test_flatten_game_builds_game_record() -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=sample_game("Alice", "Bob", 1500, 1400, "win", "resigned"),
    )

    assert record.white_username == "alice"
    assert record.black_username == "bob"
    assert record.target == "white_win"
    assert record.pgn_result == "1-0"
    assert record.termination == "Alice won by resignation"


def test_build_modeling_rows_minimal() -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=2,
        group_number=3,
        game=sample_game("Alice", "Bob", 1500, 1400, "win", "resigned"),
    )

    rows = build_modeling_rows([record])

    assert len(rows) == 1
    row = rows[0]
    assert set(row) == MODELING_COLUMNS
    assert row["event"] == "sample"
    assert row["round"] == 2
    assert row["group"] == 3
    assert row["white_username"] == "alice"
    assert row["black_username"] == "bob"
    assert row["white_rating"] == 1500
    assert row["black_rating"] == 1400
    assert row["target"] == "white_win"


def test_modeling_rows_are_leak_free() -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=sample_game("Alice", "Bob", 1500, 1400, "win", "resigned"),
    )

    row = build_modeling_rows([record])[0]

    assert set(row) == MODELING_COLUMNS
    assert LEAKY_FIELDS.isdisjoint(row)


def test_build_games_rows_keeps_audit_fields() -> None:
    record = flatten_game(
        event="sample",
        tournament_url="https://example.test/tournament",
        tournament_start_time=100_000,
        round_number=1,
        group_number=1,
        game=sample_game("Alice", "Bob", 1500, 1400, "win", "resigned"),
    )

    row = build_games_rows([record])[0]

    assert row["target"] == "white_win"
    assert row["fen"] == "8/8/8/8/8/8/8/8 w - - 0 1"
    assert row["pgn_result"] == "1-0"
    assert row["termination"] == "Alice won by resignation"


def sample_game(
    white_username: str,
    black_username: str,
    white_rating: int,
    black_rating: int,
    white_result: str,
    black_result: str,
) -> dict[str, object]:
    result = result_tag(white_result, black_result)
    return {
        "url": f"https://example.test/{white_username}-vs-{black_username}",
        "pgn": (
            '[Event "Live Chess"]\n'
            f'[White "{white_username}"]\n'
            f'[Black "{black_username}"]\n'
            f'[Result "{result}"]\n'
            '[Termination "Alice won by resignation"]\n\n'
            f"1. e4 {result}\n"
        ),
        "time_control": "300",
        "end_time": 123_456,
        "rated": True,
        "fen": "8/8/8/8/8/8/8/8 w - - 0 1",
        "time_class": "blitz",
        "rules": "chess",
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
        "tournament": "https://example.test/tournament",
    }


def result_tag(white_result: str, black_result: str) -> str:
    if white_result == "win":
        return "1-0"
    if black_result == "win":
        return "0-1"
    return "1/2-1/2"
