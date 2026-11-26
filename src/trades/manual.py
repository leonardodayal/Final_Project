"""Build in-memory roster and trade CSVs from manual player selections."""

import csv
from io import StringIO

import polars as pl

from src.trades.values import normalize_position


def build_manual_csv_inputs(
    player_values: pl.DataFrame,
    team_a: str,
    team_a_player_ids: list[str],
    team_b: str,
    team_b_player_ids: list[str],
    team_a_sends: list[str],
    team_b_sends: list[str],
) -> tuple[StringIO, StringIO]:
    """Validate selected players and serialize rosters/trade in memory."""
    team_a = team_a.strip()
    team_b = team_b.strip()
    if not team_a or not team_b:
        raise ValueError("Both fantasy team names are required.")
    if team_a == team_b:
        raise ValueError("Fantasy team names must be different.")
    if not team_a_player_ids or not team_b_player_ids:
        raise ValueError("Select at least one player for each team.")
    if not team_a_sends or not team_b_sends:
        raise ValueError("Both teams must send at least one player.")
    if len(set(team_a_player_ids)) != len(team_a_player_ids):
        raise ValueError(f"{team_a} has a player selected more than once.")
    if len(set(team_b_player_ids)) != len(team_b_player_ids):
        raise ValueError(f"{team_b} has a player selected more than once.")
    if set(team_a_player_ids) & set(team_b_player_ids):
        raise ValueError("A player cannot appear on both fantasy rosters.")
    if len(set(team_a_sends)) != len(team_a_sends) or len(
        set(team_b_sends)
    ) != len(team_b_sends):
        raise ValueError("A player cannot be included in the trade more than once.")
    if not set(team_a_sends).issubset(team_a_player_ids):
        raise ValueError(f"{team_a} can only trade players on its roster.")
    if not set(team_b_sends).issubset(team_b_player_ids):
        raise ValueError(f"{team_b} can only trade players on its roster.")

    required = {"player_id", "player_name", "position"}
    missing = required - set(player_values.columns)
    if missing:
        raise ValueError(
            "Player values are missing columns: " + ", ".join(sorted(missing))
        )
    catalog = {
        str(row["player_id"]): row
        for row in player_values.select(*required).to_dicts()
    }
    selected_ids = set(team_a_player_ids) | set(team_b_player_ids)
    unknown_ids = selected_ids - set(catalog)
    if unknown_ids:
        raise ValueError("A selected player is not in the forecast player pool.")

    roster = StringIO()
    roster_writer = csv.writer(roster)
    roster_writer.writerow(["team", "player_name", "position"])
    for team, player_ids in (
        (team_a, team_a_player_ids),
        (team_b, team_b_player_ids),
    ):
        for player_id in player_ids:
            player = catalog[player_id]
            roster_writer.writerow(
                [
                    team,
                    player["player_name"],
                    normalize_position(player["position"]),
                ]
            )
    roster.seek(0)

    trade = StringIO()
    trade_writer = csv.writer(trade)
    trade_writer.writerow(["from_team", "player_name"])
    for team, player_ids in ((team_a, team_a_sends), (team_b, team_b_sends)):
        for player_id in player_ids:
            trade_writer.writerow([team, catalog[player_id]["player_name"]])
    trade.seek(0)
    return roster, trade
