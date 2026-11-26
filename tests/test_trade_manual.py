import polars as pl
import pytest

from src.trades.manual import build_manual_csv_inputs


def _player_values() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": ["a-1", "a-2", "b-1", "b-2"],
            "player_name": ["Alpha", "Ally", "Beta", "Brett"],
            "position": ["QB", "WR", "RB", "TE"],
        }
    )


def test_build_manual_inputs_serializes_rosters_and_trade() -> None:
    roster, trade = build_manual_csv_inputs(
        _player_values(),
        " Team A ",
        ["a-1", "a-2"],
        "Team B",
        ["b-1", "b-2"],
        ["a-2"],
        ["b-1"],
    )

    assert roster.read() == (
        "team,player_name,position\r\n"
        "Team A,Alpha,QB\r\n"
        "Team A,Ally,WR\r\n"
        "Team B,Beta,RB\r\n"
        "Team B,Brett,TE\r\n"
    )
    assert trade.read() == (
        "from_team,player_name\r\nTeam A,Ally\r\nTeam B,Beta\r\n"
    )


def test_build_manual_inputs_rejects_players_not_on_rosters() -> None:
    with pytest.raises(ValueError, match="only trade players"):
        build_manual_csv_inputs(
            _player_values(),
            "Team A",
            ["a-1"],
            "Team B",
            ["b-1"],
            ["b-1"],
            ["b-1"],
        )


def test_build_manual_inputs_rejects_unknown_forecast_player() -> None:
    with pytest.raises(ValueError, match="not in the forecast player pool"):
        build_manual_csv_inputs(
            _player_values(),
            "Team A",
            ["missing"],
            "Team B",
            ["b-1"],
            ["missing"],
            ["b-1"],
        )
