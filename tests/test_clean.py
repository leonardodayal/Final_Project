import polars as pl
import pytest

from src.data.clean import build_player_week_dataset, calculate_player_scores


def _player_stats_row(**overrides) -> dict:
    row = {
        "season": 2025,
        "week": 1,
        "season_type": "REG",
        "player_id": "player-1",
        "player_name": "Test Player",
        "player_display_name": "Test Player",
        "position": "RB",
        "team": "BUF",
        "fantasy_points_ppr": 0.0,
        "passing_yards": 0,
        "passing_tds": 0,
        "passing_interceptions": 0,
        "passing_2pt_conversions": 0,
        "rushing_yards": 0,
        "rushing_tds": 0,
        "rushing_2pt_conversions": 0,
        "receptions": 0,
        "receiving_yards": 0,
        "receiving_tds": 0,
        "receiving_2pt_conversions": 0,
        "fumbles_lost_total": 0,
        "special_teams_tds": 0,
        "fumble_recovery_tds": 0,
        "fg_made_0_19": 0,
        "fg_made_20_29": 0,
        "fg_made_30_39": 0,
        "fg_made_40_49": 0,
        "fg_made_50_59": 0,
        "fg_made_60_": 0,
        "fg_missed": 0,
        "pat_made": 0,
    }
    row.update(overrides)
    return row


def _team_stats_row(team: str, opponent: str, game_id: str, **overrides) -> dict:
    row = {
        "season": 2025,
        "week": 1,
        "season_type": "REG",
        "game_id": game_id,
        "team": team,
        "opponent_team": opponent,
        "def_sacks": 0,
        "def_interceptions": 0,
        "fumble_recovery_opp": 0,
        "def_tds": 0,
        "special_teams_tds": 0,
        "def_safeties": 0,
        "def_punt_blocks": 0,
        "def_pat_blocks": 0,
        "def_fg_blocks": 0,
    }
    row.update(overrides)
    return row


def test_calculate_player_scores_applies_full_ppr_and_turnover_rules() -> None:
    stats = pl.DataFrame(
        [
            _player_stats_row(
                rushing_yards=80,
                receptions=2,
                receiving_yards=20,
                receiving_tds=1,
                fumbles_lost_total=1,
                fumble_recovery_tds=1,
            )
        ]
    )

    scored = calculate_player_scores(stats)

    assert scored["full_ppr_points"].item() == pytest.approx(22.0)


def test_calculate_player_scores_uses_kicker_distance_buckets() -> None:
    stats = pl.DataFrame(
        [
            _player_stats_row(
                position="K",
                fg_made_0_19=1,
                fg_made_30_39=1,
                fg_made_40_49=1,
                fg_made_50_59=1,
                fg_made_60_=1,
                fg_missed=1,
                pat_made=2,
            )
        ]
    )

    scored = calculate_player_scores(stats)

    assert scored["full_ppr_points"].item() == pytest.approx(22.0)


def test_build_dataset_fills_roster_zero_and_d_st_bye_weeks() -> None:
    stats = pl.DataFrame(
        [
            _player_stats_row(
                player_id="played",
                player_name="Played",
                player_display_name="Played",
            ),
            _player_stats_row(
                player_id="dst-opponent",
                player_name="Opponent",
                player_display_name="Opponent",
                position="QB",
                team="MIA",
                fantasy_points_ppr=0.0,
            ),
        ]
    )
    rosters = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "gsis_id": "played",
                "position": "RB",
                "status": "ACT",
                "team": "BUF",
                "full_name": "Played",
            },
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "gsis_id": "inactive",
                "position": "WR",
                "status": "INA",
                "team": "BUF",
                "full_name": "Inactive",
            },
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_stats_row(
                "BUF",
                "MIA",
                "game-1",
                def_sacks=1,
                def_interceptions=1,
            ),
            _team_stats_row("MIA", "BUF", "game-1"),
            _team_stats_row("KC", "PHI", "game-2", week=2),
            _team_stats_row("PHI", "KC", "game-2", week=2),
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "game_id": "game-1",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 20,
                "away_score": 10,
            },
            {
                "season": 2025,
                "week": 2,
                "game_type": "REG",
                "game_id": "game-2",
                "home_team": "KC",
                "away_team": "PHI",
                "home_score": 17,
                "away_score": 13,
            },
        ]
    )

    dataset = build_player_week_dataset(stats, rosters, team_stats, schedules)
    inactive = dataset.filter(pl.col("player_id") == "inactive").row(0, named=True)
    bills_dst = dataset.filter(pl.col("player_id") == "DST:BUF")

    assert inactive["full_ppr_points"] == 0
    assert inactive["stats_observed"] is False
    assert inactive["score_source"] == "roster_zero"
    assert bills_dst.filter(pl.col("week") == 1)["full_ppr_points"].item() == 6
    assert bills_dst.filter(pl.col("week") == 2)["full_ppr_points"].item() == 0
    assert bills_dst.filter(pl.col("week") == 2)["is_bye_week"].item() is True
    assert dataset.filter(pl.col("season_type") == "POST").is_empty()


def test_dst_points_allowed_excludes_opponent_defensive_touchdowns() -> None:
    stats = pl.DataFrame([_player_stats_row()])
    rosters = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "game_type": pl.String,
            "gsis_id": pl.String,
            "position": pl.String,
            "status": pl.String,
            "team": pl.String,
            "full_name": pl.String,
        }
    )
    team_stats = pl.DataFrame(
        [
            _team_stats_row("BUF", "MIA", "game-1", def_sacks=1),
            _team_stats_row("MIA", "BUF", "game-1", def_tds=1),
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2025,
                "week": 1,
                "game_type": "REG",
                "game_id": "game-1",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 20,
                "away_score": 27,
            }
        ]
    )

    dataset = build_player_week_dataset(stats, rosters, team_stats, schedules)
    bills_dst = dataset.filter(pl.col("player_id") == "DST:BUF").row(0, named=True)

    assert bills_dst["points_allowed"] == 21
    assert bills_dst["full_ppr_points"] == 1


def test_cancelled_game_zero_is_not_labeled_as_a_bye() -> None:
    stats = pl.DataFrame([_player_stats_row(season=2022, week=16)])
    rosters = pl.DataFrame(
        [
            {
                "season": 2022,
                "week": week,
                "game_type": "REG",
                "gsis_id": "player-1",
                "position": "RB",
                "status": "ACT",
                "team": "BUF",
                "full_name": "Test Player",
            }
            for week in (16, 17)
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_stats_row("BUF", "CIN", "game-16", season=2022, week=16),
            _team_stats_row("CIN", "BUF", "game-16", season=2022, week=16),
            _team_stats_row("KC", "PHI", "game-17", season=2022, week=17),
            _team_stats_row("PHI", "KC", "game-17", season=2022, week=17),
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2022,
                "week": 16,
                "game_type": "REG",
                "game_id": "game-16",
                "home_team": "BUF",
                "away_team": "CIN",
                "home_score": 20,
                "away_score": 10,
            },
            {
                "season": 2022,
                "week": 17,
                "game_type": "REG",
                "game_id": "game-17",
                "home_team": "KC",
                "away_team": "PHI",
                "home_score": 17,
                "away_score": 13,
            },
        ]
    )

    dataset = build_player_week_dataset(stats, rosters, team_stats, schedules)
    bills_player = dataset.filter(
        (pl.col("player_id") == "player-1") & (pl.col("week") == 17)
    ).row(0, named=True)
    bills_dst = dataset.filter(
        (pl.col("player_id") == "DST:BUF") & (pl.col("week") == 17)
    ).row(0, named=True)

    for row in (bills_player, bills_dst):
        assert row["full_ppr_points"] == 0
        assert row["is_bye_week"] is False
        assert row["is_cancelled_game_week"] is True
    assert bills_dst["score_source"] == "cancelled_game_zero"
