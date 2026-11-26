"""Evaluate the effect of a proposed two-team fantasy trade."""

import argparse
import csv
import json
from pathlib import Path
from typing import TextIO

import polars as pl

from src.trades.roster import (
    BENCH_WEIGHT,
    LineupPlayer,
    RosterPlayer,
    assign_lineup,
    match_roster,
    normalize_player_name,
    transfer_player,
)
from src.trades.values import PROJECTION_COLUMN, aggregate_rest_of_season_values

TRADE_COLUMNS = {"from_team", "player_name"}


def _read_trade(source: Path | TextIO) -> list[dict[str, str]]:
    if isinstance(source, Path):
        with source.open(newline="", encoding="utf-8-sig") as file:
            return _parse_trade_file(file)
    return _parse_trade_file(source)


def _parse_trade_file(file: TextIO) -> list[dict[str, str]]:
    reader = csv.DictReader(file)
    if reader.fieldnames is None or not TRADE_COLUMNS.issubset(reader.fieldnames):
        raise ValueError("Trade CSV must contain from_team and player_name columns.")
    rows = []
    for line_number, row in enumerate(reader, start=2):
        team = (row.get("from_team") or "").strip()
        name = (row.get("player_name") or "").strip()
        if not team or not name:
            raise ValueError(
                f"Trade CSV row {line_number} has an empty required value."
            )
        rows.append({"from_team": team, "player_name": name})
    if not rows:
        raise ValueError("Trade CSV contains no player transfers.")
    return rows


def _team_report(
    team: str,
    players: list[RosterPlayer],
    bench_weight: float,
) -> dict:
    lineup = assign_lineup(players, bench_weight)
    strength = sum(player.weighted_value for player in lineup)
    return {
        "team": team,
        "strength": strength,
        "starters": [
            _lineup_entry(entry)
            for entry in lineup
            if entry.is_starter
        ],
        "bench": [
            _lineup_entry(entry)
            for entry in lineup
            if not entry.is_starter
        ],
    }


def _lineup_entry(entry: LineupPlayer) -> dict:
    return {
        "slot": entry.slot,
        "player_name": entry.player.player_name,
        "position": entry.player.position,
        "ros_value": entry.player.ros_value,
        "weighted_value": entry.weighted_value,
    }


def evaluate_trade(
    predictions: pl.DataFrame,
    roster_path: Path | TextIO,
    trade_path: Path | TextIO,
    bench_weight: float = BENCH_WEIGHT,
) -> dict:
    """Return before/after team strengths and the trade classification."""
    if not 0.0 <= bench_weight <= 1.0:
        raise ValueError("Bench weight must be between 0 and 1.")
    values = aggregate_rest_of_season_values(predictions)
    rosters = match_roster(roster_path, values)
    if len(rosters) != 2:
        raise ValueError("Roster CSV must contain exactly two fantasy teams.")
    teams = sorted(rosters)

    player_by_team_and_name = {}
    for team, players in rosters.items():
        for player in players:
            key = (team, normalize_player_name(player.player_name))
            if key in player_by_team_and_name:
                raise ValueError(
                    f"{player.player_name} is not unique on {team}'s roster."
                )
            player_by_team_and_name[key] = player
    transfers = _read_trade(trade_path)
    outgoing_ids: set[str] = set()
    outgoing_by_team: dict[str, list[RosterPlayer]] = {team: [] for team in teams}
    for transfer in transfers:
        sender = transfer["from_team"]
        if sender not in rosters:
            raise ValueError(f"Unknown trade team: {sender!r}.")
        key = (sender, normalize_player_name(transfer["player_name"]))
        player = player_by_team_and_name.get(key)
        if player is None:
            raise ValueError(
                f"{transfer['player_name']} is not on {sender}'s roster."
            )
        if player.player_id in outgoing_ids:
            raise ValueError(
                f"{player.player_name} appears more than once in the trade."
            )
        outgoing_ids.add(player.player_id)
        outgoing_by_team[sender].append(player)

    if any(not outgoing_by_team[team] for team in teams):
        raise ValueError("Both teams must send at least one player in the trade.")

    after = {team: list(players) for team, players in rosters.items()}
    for sender, receiver in ((teams[0], teams[1]), (teams[1], teams[0])):
        for player in outgoing_by_team[sender]:
            after[sender].remove(player)
            after[receiver].append(transfer_player(player, receiver))

    before_reports = {
        team: _team_report(team, rosters[team], bench_weight)
        for team in teams
    }
    after_reports = {
        team: _team_report(team, after[team], bench_weight)
        for team in teams
    }
    deltas = {
        team: after_reports[team]["strength"] - before_reports[team]["strength"]
        for team in teams
    }
    improves = {team: deltas[team] > 0 for team in teams}
    if all(improves.values()):
        classification = "mutually_beneficial"
    elif all(delta == 0 for delta in deltas.values()):
        classification = "neutral"
    elif any(improves.values()):
        classification = "one_sided"
    else:
        classification = "neither_team_improves"

    return {
        "season": int(values["season"][0]),
        "forecast_weeks": sorted(int(week) for week in predictions["week"].unique()),
        "projection_column": PROJECTION_COLUMN,
        "bench_weight": bench_weight,
        "trade_classification": classification,
        "mutually_beneficial": all(improves.values()),
        "uncertainty_note": (
            "Model uncertainty is not estimated; small positive deltas may not "
            "represent meaningful improvements."
        ),
        "teams": [
            {
                "team": team,
                "before": before_reports[team],
                "after": after_reports[team],
                "delta": deltas[team],
                "percent_change": (
                    100 * deltas[team] / before_reports[team]["strength"]
                    if before_reports[team]["strength"] != 0
                    else None
                ),
                "improves": improves[team],
            }
            for team in teams
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--rosters", type=Path, required=True)
    parser.add_argument("--trade", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bench-weight",
        type=float,
        default=BENCH_WEIGHT,
        help="Bench value multiplier between 0 and 1 (default: 2/3).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_trade(
        pl.read_parquet(args.predictions),
        args.rosters,
        args.trade,
        args.bench_weight,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"trade_classification: {report['trade_classification']}")
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
