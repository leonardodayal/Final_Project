import json

import polars as pl
import pytest

from src.trades.evaluate import evaluate_trade, main
from src.trades.roster import (
    LineupPlayer,
    RosterPlayer,
    assign_lineup,
    match_roster,
)
from src.trades.uploads import read_csv_upload
from src.trades.values import aggregate_rest_of_season_values, main as values_main


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "season": 2025,
                "week": week,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "predicted_full_ppr_points": -100.0,
                "predicted_full_ppr_points_if_active": score,
            }
            for player_id, player_name, position, scores in (
                ("qb-a", "Quarterback Alpha", "QB", (10.0, 10.0)),
                ("rb-a", "Runner Alpha", "RB", (20.0, 20.0)),
                ("qb-b", "Quarterback Beta", "QB", (12.0, 12.0)),
                ("rb-b", "Runner Beta", "RB", (10.0, 10.0)),
            )
            for week, score in enumerate(scores, start=1)
        ]
    )


def test_aggregate_ros_uses_if_active_scores_without_model_role_weighting() -> None:
    values = aggregate_rest_of_season_values(_predictions())

    alpha = values.filter(pl.col("player_id") == "rb-a").row(0, named=True)
    assert alpha["ros_value"] == 40.0
    assert alpha["forecast_weeks"] == 2


def test_aggregate_ros_rejects_duplicate_player_week_rows() -> None:
    predictions = _predictions()
    predictions = pl.concat([predictions, predictions.head(1)])

    with pytest.raises(ValueError, match="duplicate player-week"):
        aggregate_rest_of_season_values(predictions)


def test_values_cli_writes_rest_of_season_csv(tmp_path) -> None:
    predictions_path = tmp_path / "weekly.parquet"
    _predictions().write_parquet(predictions_path)
    output_path = tmp_path / "processed" / "values.csv"

    result = values_main(
        [
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    values = pl.read_csv(output_path)
    alpha = values.filter(pl.col("player_id") == "rb-a").row(0, named=True)
    assert alpha["ros_value"] == 40.0


def test_roster_matching_rejects_ambiguous_names(tmp_path) -> None:
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "team,player_name,position\nTeam A,Runner Alpha,RB\n",
        encoding="utf-8",
    )
    values = pl.DataFrame(
        {
            "player_id": ["rb-a", "rb-b"],
            "player_name": ["Runner Alpha", "Runner Alpha"],
            "position": ["RB", "RB"],
            "ros_value": [20.0, 15.0],
        }
    )

    with pytest.raises(ValueError, match="ambiguous"):
        match_roster(roster, values)


def test_lineup_uses_flex_for_highest_remaining_rb_wr_or_te() -> None:
    players = [
        RosterPlayer("Team A", "Quarterback", "QB", "qb", 40.0),
        RosterPlayer("Team A", "Runner 1", "RB", "rb1", 10.0),
        RosterPlayer("Team A", "Runner 2", "RB", "rb2", 9.0),
        RosterPlayer("Team A", "Runner 3", "RB", "rb3", 8.0),
        RosterPlayer("Team A", "Receiver 1", "WR", "wr1", 30.0),
        RosterPlayer("Team A", "Receiver 2", "WR", "wr2", 20.0),
        RosterPlayer("Team A", "Receiver 3", "WR", "wr3", 19.0),
        RosterPlayer("Team A", "Tight End 1", "TE", "te1", 15.0),
        RosterPlayer("Team A", "Tight End 2", "TE", "te2", 5.0),
        RosterPlayer("Team A", "Kicker", "K", "k", 3.0),
        RosterPlayer("Team A", "Defense", "DST", "dst", 4.0),
    ]

    lineup = assign_lineup(players)

    flex = next(entry for entry in lineup if entry.slot == "FLEX")
    bench = [entry for entry in lineup if entry.slot == "BENCH"]
    assert isinstance(flex, LineupPlayer)
    assert flex.player.player_id == "wr3"
    assert {entry.player.player_id for entry in bench} == {"rb3", "te2"}
    assert sum(entry.weighted_value for entry in lineup) == pytest.approx(
        40 + 10 + 9 + 30 + 20 + 15 + 19 + 3 + 4 + (8 + 5) * (2 / 3)
    )

    half_weight_lineup = assign_lineup(players, bench_weight=0.5)
    assert sum(entry.weighted_value for entry in half_weight_lineup) == pytest.approx(
        150 + (8 + 5) * 0.5
    )


def test_trade_evaluation_reports_one_sided_result(tmp_path) -> None:
    roster = tmp_path / "rosters.csv"
    roster.write_text(
        "team,player_name,position\n"
        "Team A,Quarterback Alpha,QB\n"
        "Team A,Runner Alpha,RB\n"
        "Team B,Quarterback Beta,QB\n"
        "Team B,Runner Beta,RB\n",
        encoding="utf-8",
    )
    trade = tmp_path / "trade.csv"
    trade.write_text(
        "from_team,player_name\nTeam A,Runner Alpha\nTeam B,Runner Beta\n",
        encoding="utf-8",
    )

    report = evaluate_trade(_predictions(), roster, trade)

    assert report["trade_classification"] == "one_sided"
    assert report["mutually_beneficial"] is False
    assert report["teams"][0]["delta"] == -20.0
    assert report["teams"][1]["delta"] == 20.0
    assert report["projection_column"] == "predicted_full_ppr_points_if_active"


def test_trade_evaluation_accepts_in_memory_csv_uploads() -> None:
    roster = read_csv_upload(
        b"team,player_name,position\n"
        b"Team A,Quarterback Alpha,QB\n"
        b"Team A,Runner Alpha,RB\n"
        b"Team B,Quarterback Beta,QB\n"
        b"Team B,Runner Beta,RB\n"
    )
    trade = read_csv_upload(
        b"from_team,player_name\nTeam A,Runner Alpha\nTeam B,Runner Beta\n"
    )

    report = evaluate_trade(_predictions(), roster, trade)

    assert report["trade_classification"] == "one_sided"


def test_trade_evaluation_detects_mutual_lineup_improvement(tmp_path) -> None:
    player_data = [
        ("a-qb", "A Quarterback", "QB", "Team A", 10.0),
        ("a-rb1", "A Runner 1", "RB", "Team A", 10.0),
        ("a-rb2", "A Runner 2", "RB", "Team A", 9.0),
        ("a-rb3", "A Runner 3", "RB", "Team A", 20.0),
        ("a-wr1", "A Receiver 1", "WR", "Team A", 5.0),
        ("a-wr2", "A Receiver 2", "WR", "Team A", 4.0),
        ("a-te1", "A Tight End 1", "TE", "Team A", 30.0),
        ("a-te2", "A Tight End 2", "TE", "Team A", 29.0),
        ("b-qb", "B Quarterback", "QB", "Team B", 10.0),
        ("b-rb1", "B Runner 1", "RB", "Team B", 5.0),
        ("b-rb2", "B Runner 2", "RB", "Team B", 4.0),
        ("b-wr1", "B Receiver 1", "WR", "Team B", 40.0),
        ("b-wr2", "B Receiver 2", "WR", "Team B", 35.0),
        ("b-wr3", "B Receiver 3", "WR", "Team B", 20.0),
        ("b-te1", "B Tight End 1", "TE", "Team B", 50.0),
        ("b-te2", "B Tight End 2", "TE", "Team B", 40.0),
    ]
    predictions = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": week,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "predicted_full_ppr_points_if_active": value / 2,
            }
            for player_id, player_name, position, _, value in player_data
            for week in (1, 2)
        ]
    )
    roster = tmp_path / "rosters.csv"
    roster.write_text(
        "team,player_name,position\n"
        + "".join(
            f"{team},{name},{position}\n"
            for _, name, position, team, _ in player_data
        ),
        encoding="utf-8",
    )
    trade = tmp_path / "trade.csv"
    trade.write_text(
        "from_team,player_name\nTeam A,A Runner 3\nTeam B,B Receiver 3\n",
        encoding="utf-8",
    )

    report = evaluate_trade(predictions, roster, trade)

    assert report["trade_classification"] == "mutually_beneficial"
    assert all(team["delta"] > 0 for team in report["teams"])


def test_trade_evaluation_classifies_equal_exchange_as_neutral(tmp_path) -> None:
    predictions = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": week,
                "player_id": player_id,
                "player_name": player_name,
                "position": "RB",
                "predicted_full_ppr_points_if_active": 10.0,
            }
            for player_id, player_name in (("rb-a", "Runner A"), ("rb-b", "Runner B"))
            for week in (1, 2)
        ]
    )
    roster = tmp_path / "rosters.csv"
    roster.write_text(
        "team,player_name,position\nTeam A,Runner A,RB\nTeam B,Runner B,RB\n",
        encoding="utf-8",
    )
    trade = tmp_path / "trade.csv"
    trade.write_text(
        "from_team,player_name\nTeam A,Runner A\nTeam B,Runner B\n",
        encoding="utf-8",
    )

    report = evaluate_trade(predictions, roster, trade)

    assert report["trade_classification"] == "neutral"
    assert [team["delta"] for team in report["teams"]] == [0.0, 0.0]


def test_trade_evaluation_reports_when_neither_team_improves(tmp_path) -> None:
    player_data = [
        ("a-rb1", "A Runner 1", "RB", "Team A", 30.0),
        ("a-rb2", "A Runner 2", "RB", "Team A", 25.0),
        ("a-wr1", "A Receiver 1", "WR", "Team A", 40.0),
        ("a-wr2", "A Receiver 2", "WR", "Team A", 35.0),
        ("a-te1", "A Tight End 1", "TE", "Team A", 50.0),
        ("a-te2", "A Tight End 2", "TE", "Team A", 40.0),
        ("b-rb1", "B Runner 1", "RB", "Team B", 40.0),
        ("b-rb2", "B Runner 2", "RB", "Team B", 35.0),
        ("b-wr1", "B Receiver 1", "WR", "Team B", 30.0),
        ("b-wr2", "B Receiver 2", "WR", "Team B", 25.0),
        ("b-te1", "B Tight End 1", "TE", "Team B", 50.0),
        ("b-te2", "B Tight End 2", "TE", "Team B", 40.0),
    ]
    predictions = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": week,
                "player_id": player_id,
                "player_name": player_name,
                "position": position,
                "predicted_full_ppr_points_if_active": value / 2,
            }
            for player_id, player_name, position, _, value in player_data
            for week in (1, 2)
        ]
    )
    roster = tmp_path / "rosters.csv"
    roster.write_text(
        "team,player_name,position\n"
        + "".join(
            f"{team},{name},{position}\n"
            for _, name, position, team, _ in player_data
        ),
        encoding="utf-8",
    )
    trade = tmp_path / "trade.csv"
    trade.write_text(
        "from_team,player_name\nTeam A,A Runner 1\nTeam B,B Receiver 1\n",
        encoding="utf-8",
    )

    report = evaluate_trade(predictions, roster, trade)

    assert report["trade_classification"] == "neither_team_improves"
    assert all(team["delta"] < 0 for team in report["teams"])


def test_trade_cli_writes_a_json_report(tmp_path) -> None:
    predictions_path = tmp_path / "weekly.parquet"
    _predictions().write_parquet(predictions_path)
    roster_path = tmp_path / "rosters.csv"
    roster_path.write_text(
        "team,player_name,position\n"
        "Team A,Quarterback Alpha,QB\n"
        "Team A,Runner Alpha,RB\n"
        "Team B,Quarterback Beta,QB\n"
        "Team B,Runner Beta,RB\n",
        encoding="utf-8",
    )
    trade_path = tmp_path / "trade.csv"
    trade_path.write_text(
        "from_team,player_name\nTeam A,Runner Alpha\nTeam B,Runner Beta\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "reports" / "trade.json"

    result = main(
        [
            "--predictions",
            str(predictions_path),
            "--rosters",
            str(roster_path),
            "--trade",
            str(trade_path),
            "--output",
            str(output_path),
        ]
    )

    assert result == 0
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["trade_classification"] == "one_sided"
