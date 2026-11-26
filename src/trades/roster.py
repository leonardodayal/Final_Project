"""Parse fantasy rosters and assign starters under the project lineup rules."""

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TextIO

import polars as pl

from src.trades.values import normalize_position

STARTER_SLOTS = {
    "QB": 1,
    "RB": 2,
    "WR": 2,
    "TE": 1,
    "DST": 1,
    "K": 1,
}
FLEX_POSITIONS = {"RB", "WR", "TE"}
BENCH_WEIGHT = 2 / 3
ROSTER_COLUMNS = {"team", "player_name", "position"}


@dataclass(frozen=True)
class RosterPlayer:
    team: str
    player_name: str
    position: str
    player_id: str
    ros_value: float


@dataclass(frozen=True)
class LineupPlayer:
    player: RosterPlayer
    slot: str
    bench_weight: float = BENCH_WEIGHT

    @property
    def is_starter(self) -> bool:
        return self.slot != "BENCH"

    @property
    def weighted_value(self) -> float:
        if self.is_starter:
            return self.player.ros_value
        return self.player.ros_value * self.bench_weight


def normalize_player_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _read_roster_rows(source: Path | TextIO) -> list[dict[str, str]]:
    if isinstance(source, Path):
        with source.open(newline="", encoding="utf-8-sig") as file:
            return _parse_roster_file(file)
    return _parse_roster_file(source)


def _parse_roster_file(file: TextIO) -> list[dict[str, str]]:
    reader = csv.DictReader(file)
    if reader.fieldnames is None or not ROSTER_COLUMNS.issubset(reader.fieldnames):
        raise ValueError(
            "Roster CSV must contain team, player_name, and position columns."
        )
    rows = []
    for line_number, row in enumerate(reader, start=2):
        team = (row.get("team") or "").strip()
        name = (row.get("player_name") or "").strip()
        position = (row.get("position") or "").strip()
        if not team or not name or not position:
            raise ValueError(
                f"Roster CSV row {line_number} has an empty required value."
            )
        try:
            normalized_position = normalize_position(position)
        except ValueError as error:
            raise ValueError(f"Roster CSV row {line_number}: {error}") from error
        rows.append(
            {
                "team": team,
                "player_name": name,
                "position": normalized_position,
            }
        )
    if not rows:
        raise ValueError("Roster CSV contains no players.")
    return rows


def match_roster(
    roster_source: Path | TextIO,
    player_values: pl.DataFrame,
) -> dict[str, list[RosterPlayer]]:
    """Resolve roster names against forecasts; reject missing/ambiguous matches."""
    required = {"player_id", "player_name", "position", "ros_value"}
    missing = required - set(player_values.columns)
    if missing:
        raise ValueError(
            "Player values are missing columns: " + ", ".join(sorted(missing))
        )
    by_name_position: dict[tuple[str, str], list[dict]] = {}
    for player in player_values.select(*required).to_dicts():
        key = (
            normalize_player_name(player["player_name"]),
            normalize_position(player["position"]),
        )
        by_name_position.setdefault(key, []).append(player)

    teams: dict[str, list[RosterPlayer]] = {}
    roster_player_ids: set[str] = set()
    for row in _read_roster_rows(roster_source):
        key = (normalize_player_name(row["player_name"]), row["position"])
        matches = by_name_position.get(key, [])
        if not matches:
            raise ValueError(
                f"No forecast found for {row['player_name']} ({row['position']})."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Forecast match for {row['player_name']} ({row['position']}) "
                "is ambiguous; use a unique player name."
            )
        match = matches[0]
        player_id = str(match["player_id"])
        if player_id in roster_player_ids:
            raise ValueError(
                f"{row['player_name']} appears more than once in the rosters."
            )
        roster_player_ids.add(player_id)
        teams.setdefault(row["team"], []).append(
            RosterPlayer(
                team=row["team"],
                player_name=str(match["player_name"]),
                position=row["position"],
                player_id=player_id,
                ros_value=float(match["ros_value"]),
            )
        )
    return teams


def _player_order(player: RosterPlayer) -> tuple[float, str, str]:
    return (-player.ros_value, player.player_name.casefold(), player.player_id)


def assign_lineup(
    players: list[RosterPlayer],
    bench_weight: float = BENCH_WEIGHT,
) -> list[LineupPlayer]:
    """Fill fixed slots, then one RB/WR/TE flex slot, maximizing ROS value."""
    if not 0.0 <= bench_weight <= 1.0:
        raise ValueError("Bench weight must be between 0 and 1.")
    assigned: dict[str, str] = {}
    for position, count in STARTER_SLOTS.items():
        candidates = sorted(
            (player for player in players if player.position == position),
            key=_player_order,
        )
        for index, player in enumerate(candidates[:count], start=1):
            slot = position if count == 1 else f"{position}{index}"
            assigned[player.player_id] = slot

    flex_candidates = sorted(
        (
            player
            for player in players
            if player.position in FLEX_POSITIONS
            and player.player_id not in assigned
        ),
        key=_player_order,
    )
    if flex_candidates:
        assigned[flex_candidates[0].player_id] = "FLEX"

    return [
        LineupPlayer(
            player=player,
            slot=assigned.get(player.player_id, "BENCH"),
            bench_weight=bench_weight,
        )
        for player in sorted(players, key=_player_order)
    ]


def transfer_player(
    player: RosterPlayer,
    new_team: str,
) -> RosterPlayer:
    """Return a copy of a roster player owned by the receiving fantasy team."""
    return replace(player, team=new_team)
