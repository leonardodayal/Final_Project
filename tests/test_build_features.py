from datetime import datetime, timezone

import polars as pl
import pytest

from src.features.build import (
    FEATURE_COLUMNS,
    _injury_features,
    _rank_features,
    _week_cutoffs,
    build_feature_dataset,
)


def _player_week(week: int, score: float, **overrides) -> dict:
    row = {
        "season": 2024,
        "season_type": "REG",
        "week": week,
        "player_id": "player-1",
        "player_name": "Test Player",
        "player_display_name": "Test Player",
        "position": "RB",
        "team": "BUF",
        "opponent_team": "WRONG",
        "full_ppr_points": score,
        "stats_observed": True,
        "is_bye_week": False,
        "is_cancelled_game_week": False,
        "score_source": "nflverse_player_stats",
        "carries": week * 2,
        "targets": week * 4,
        "receptions": week,
        "attempts": 0,
        "passing_yards": 0,
        "rushing_yards": week * 10,
        "receiving_yards": week * 5,
        "points_allowed": None,
    }
    row.update(overrides)
    return row


def _dst_week(week: int, team: str, points_allowed: int) -> dict:
    return {
        "season": 2024,
        "season_type": "REG",
        "week": week,
        "player_id": f"DST:{team}",
        "player_name": f"{team} D/ST",
        "player_display_name": f"{team} D/ST",
        "position": "DST",
        "team": team,
        "opponent_team": None,
        "full_ppr_points": 5.0,
        "stats_observed": True,
        "is_bye_week": False,
        "is_cancelled_game_week": False,
        "score_source": "nflverse_team_stats",
        "carries": None,
        "targets": None,
        "receptions": None,
        "attempts": None,
        "passing_yards": None,
        "rushing_yards": None,
        "receiving_yards": None,
        "points_allowed": points_allowed,
    }


def _team_game(
    week: int,
    game_id: str,
    team: str,
    opponent: str,
    passing_yards: int,
    rushing_yards: int,
) -> dict:
    return {
        "season": 2024,
        "week": week,
        "season_type": "REG",
        "game_id": game_id,
        "team": team,
        "opponent_team": opponent,
        "passing_yards": passing_yards,
        "rushing_yards": rushing_yards,
    }


def test_features_use_only_prior_player_and_opponent_games() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_id": "buf-ari-1",
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "ARI",
                "home_score": 24,
                "away_score": 17,
            },
            {
                "season": 2024,
                "week": 1,
                "game_id": "mia-kc-1",
                "game_type": "REG",
                "home_team": "MIA",
                "away_team": "KC",
                "home_score": 21,
                "away_score": 28,
            },
            {
                "season": 2024,
                "week": 2,
                "game_id": "buf-mia-2",
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 31,
                "away_score": 10,
            },
            {
                "season": 2024,
                "week": 3,
                "game_id": "buf-ari-3",
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "ARI",
                "home_score": 27,
                "away_score": 20,
            },
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_game(1, "buf-ari-1", "BUF", "ARI", 200, 80),
            _team_game(1, "buf-ari-1", "ARI", "BUF", 150, 70),
            _team_game(1, "mia-kc-1", "MIA", "KC", 100, 50),
            _team_game(1, "mia-kc-1", "KC", "MIA", 300, 100),
            _team_game(2, "buf-mia-2", "BUF", "MIA", 900, 500),
            _team_game(2, "buf-mia-2", "MIA", "BUF", 140, 60),
            _team_game(3, "buf-ari-3", "BUF", "ARI", 250, 90),
            _team_game(3, "buf-ari-3", "ARI", "BUF", 180, 70),
        ]
    )
    player_weeks = pl.DataFrame(
        [
            _player_week(1, 10.0),
            _player_week(2, 20.0),
            _player_week(3, 30.0),
            _dst_week(1, "BUF", 17),
            _dst_week(1, "ARI", 24),
            _dst_week(1, "MIA", 28),
            _dst_week(1, "KC", 21),
            _dst_week(2, "BUF", 10),
            _dst_week(2, "MIA", 31),
            _dst_week(3, "BUF", 20),
            _dst_week(3, "ARI", 27),
        ]
    )

    features = build_feature_dataset(player_weeks, team_stats, schedules)
    weeks = features.filter(pl.col("player_id") == "player-1").sort("week")
    week_one, week_two, week_three = weeks.to_dicts()

    assert week_one["feature_prior_player_weeks"] == 0
    assert week_one["feature_previous_week_score"] is None
    assert week_one[
        "feature_opponent_defense_passing_yards_allowed_avg_last_3_games"
    ] is None

    assert week_two["opponent_team"] == "MIA"
    assert week_two["target_full_ppr_points"] == 20.0
    assert week_two["feature_previous_week_score"] == 10.0
    assert week_two["feature_score_avg_last_3_weeks"] == 10.0
    assert week_two["feature_score_avg_season_to_date"] == 10.0
    assert week_two["feature_targets_avg_last_3_weeks"] == 4.0
    assert week_two[
        "feature_opponent_defense_passing_yards_allowed_avg_last_3_games"
    ] == 300.0
    assert week_two[
        "feature_opponent_defense_rushing_yards_allowed_avg_last_3_games"
    ] == 100.0
    assert week_two[
        "feature_opponent_defense_points_allowed_avg_last_3_games"
    ] == 28.0
    assert week_two[
        "feature_opponent_offense_passing_yards_avg_last_3_games"
    ] == 100.0

    assert week_three["feature_previous_week_score"] == 20.0
    assert week_three["feature_score_avg_last_3_weeks"] == 15.0
    assert week_three["feature_score_avg_season_to_date"] == 15.0
    assert week_three[
        "feature_opponent_defense_passing_yards_allowed_avg_last_3_games"
    ] == 200.0
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert "target_full_ppr_points" not in FEATURE_COLUMNS
    assert "passing_yards" not in FEATURE_COLUMNS


def test_feature_builder_rejects_non_regular_season_rows() -> None:
    player_weeks = pl.DataFrame([_player_week(1, 10.0)]).with_columns(
        pl.lit("POST").alias("season_type")
    )
    empty_stats = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "season_type": pl.String,
            "game_id": pl.String,
            "team": pl.String,
            "opponent_team": pl.String,
            "passing_yards": pl.Int64,
            "rushing_yards": pl.Int64,
        }
    )
    empty_schedule = pl.DataFrame(
        schema={
            "season": pl.Int64,
            "week": pl.Int64,
            "game_id": pl.String,
            "game_type": pl.String,
            "home_team": pl.String,
            "away_team": pl.String,
            "home_score": pl.Int64,
            "away_score": pl.Int64,
        }
    )

    with pytest.raises(ValueError, match="regular-season"):
        build_feature_dataset(player_weeks, empty_stats, empty_schedule)


def test_player_history_carries_forward_but_season_average_resets() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_id": "buf-ari-1",
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "ARI",
                "home_score": 24,
                "away_score": 17,
            }
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_game(1, "buf-ari-1", "BUF", "ARI", 200, 80),
            _team_game(1, "buf-ari-1", "ARI", "BUF", 150, 70),
        ]
    )
    player_weeks = pl.DataFrame(
        [
            _player_week(17, 15.0, season=2023),
            _player_week(1, 20.0),
            _dst_week(1, "BUF", 17),
            _dst_week(1, "ARI", 24),
        ]
    )

    features = build_feature_dataset(
        player_weeks,
        team_stats,
        schedules,
        output_seasons=[2024],
    )
    player = features.filter(pl.col("player_id") == "player-1").row(0, named=True)

    assert features["season"].unique().to_list() == [2024]
    assert player["feature_prior_player_weeks"] == 1
    assert player["feature_previous_week_score"] == 15.0
    assert player["feature_score_avg_last_3_weeks"] == 15.0
    assert player["feature_score_avg_season_to_date"] is None


def test_missing_player_id_keeps_availability_flags_false() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 1,
                "game_id": "buf-ari-1",
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "ARI",
                "home_score": 24,
                "away_score": 17,
            }
        ]
    )
    team_stats = pl.DataFrame(
        [
            _team_game(1, "buf-ari-1", "BUF", "ARI", 200, 80),
            _team_game(1, "buf-ari-1", "ARI", "BUF", 150, 70),
        ]
    )
    player_weeks = pl.DataFrame(
        [
            _player_week(
                1,
                0.0,
                player_id=None,
                player_name="Unidentified Player",
                player_display_name="Unidentified Player",
            ),
            _dst_week(1, "BUF", 17),
            _dst_week(1, "ARI", 24),
        ]
    )

    features = build_feature_dataset(player_weeks, team_stats, schedules)
    unidentified = features.filter(pl.col("player_id").is_null()).row(
        0, named=True
    )

    assert unidentified["feature_fantasypros_rank_available"] is False
    assert unidentified["feature_injury_report_available"] is False


def test_rank_features_use_latest_same_season_snapshot_before_kickoff() -> None:
    player_weeks = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "player_id": "player-1",
                "position": "RB",
            },
            {
                "season": 2024,
                "week": 2,
                "player_id": "player-2",
                "position": "RB",
            },
            {
                "season": 2024,
                "week": 2,
                "player_id": "player-3",
                "position": "RB",
            },
            {
                "season": 2024,
                "week": 2,
                "player_id": "DST:BUF",
                "position": "DST",
            },
            {
                "season": 2024,
                "week": 2,
                "player_id": "DST:LA",
                "position": "DST",
            },
            {
                "season": 2024,
                "week": 2,
                "player_id": "DST:JAX",
                "position": "DST",
            },
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "game_type": "REG",
                "gameday": "2024-09-12",
                "gametime": "20:15",
            }
        ]
    )
    rankings = pl.DataFrame(
        [
            {
                "page_type": "weekly-rb",
                "ecr_type": "wp",
                "id": "101",
                "ecr": 3.4,
                "scrape_date": "2024-09-11",
            },
            {
                "page_type": "redraft-rb",
                "ecr_type": "rp",
                "id": "101",
                "ecr": 99.0,
                "scrape_date": "2024-09-11",
                "team": "BUF",
            },
            {
                "page_type": "weekly-rb",
                "ecr_type": "wp",
                "id": "101",
                "ecr": 1.0,
                "scrape_date": "2024-09-12",
            },
            {
                "page_type": "weekly-rb",
                "ecr_type": "wp",
                "id": "101",
                "ecr": 2.0,
                "scrape_date": "2023-12-20",
                "team": "BUF",
            },
            {
                "page_type": "redraft-rb",
                "ecr_type": "rp",
                "id": "102",
                "ecr": 12.0,
                "scrape_date": "2024-09-10",
                "team": "BUF",
            },
            {
                "page_type": "weekly-rb",
                "ecr_type": "wsf",
                "id": "103",
                "ecr": 5.0,
                "scrape_date": "2024-09-11",
                "team": "BUF",
            },
            {
                "page_type": "weekly-dst",
                "ecr_type": "wp",
                "id": "8020",
                "ecr": 2.5,
                "scrape_date": "2024-09-11",
                "team": "BUF",
            },
            {
                "page_type": "weekly-dst",
                "ecr_type": "wp",
                "id": "8280",
                "ecr": 3.5,
                "scrape_date": "2024-09-11",
                "team": "LAR",
            },
            {
                "page_type": "weekly-dst",
                "ecr_type": "wp",
                "id": "8130",
                "ecr": 4.5,
                "scrape_date": "2024-09-11",
                "team": "JAC",
            },
        ]
    )
    player_ids = pl.DataFrame(
        {
            "fantasypros_id": [101, 102, 103],
            "gsis_id": ["player-1", "player-2", "player-3"],
        }
    )

    ranked = _rank_features(
        player_weeks,
        rankings,
        player_ids,
        _week_cutoffs(schedules),
    ).to_dicts()

    ranks = {
        row["player_id"]: row["feature_fantasypros_position_rank"]
        for row in ranked
    }
    sources = {
        row["player_id"]: row["feature_fantasypros_rank_source"]
        for row in ranked
    }
    assert ranks == {
        "player-1": 3.4,
        "player-2": 12.0,
        "player-3": None,
        "DST:BUF": 2.5,
        "DST:LA": 3.5,
        "DST:JAX": 4.5,
    }
    assert sources == {
        "player-1": "weekly",
        "player-2": "redraft",
        "player-3": None,
        "DST:BUF": "weekly",
        "DST:LA": "weekly",
        "DST:JAX": "weekly",
    }


def test_injury_features_exclude_updates_at_or_after_first_kickoff() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "game_type": "REG",
                "gameday": "2024-09-12",
                "gametime": "20:15",
            }
        ]
    )
    injuries = pl.DataFrame(
        [
            {
                "season": 2024,
                "week": 2,
                "game_type": "REG",
                "gsis_id": "player-1",
                "date_modified": datetime(
                    2024, 9, 13, 0, 14, tzinfo=timezone.utc
                ),
                "report_status": "Questionable",
                "practice_status": "Limited Participation in Practice",
            },
            {
                "season": 2024,
                "week": 2,
                "game_type": "REG",
                "gsis_id": "player-1",
                "date_modified": datetime(
                    2024, 9, 13, 0, 15, tzinfo=timezone.utc
                ),
                "report_status": "Out",
                "practice_status": "Did Not Participate In Practice",
            },
        ]
    )

    injuries_as_of = _injury_features(injuries, _week_cutoffs(schedules))
    player = injuries_as_of.row(0, named=True)

    assert player["feature_injury_report_status"] == "Questionable"
    assert (
        player["feature_injury_practice_status"]
        == "Limited Participation in Practice"
    )
    assert player["feature_injury_report_available"] is True
