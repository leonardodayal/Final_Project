from datetime import datetime, timezone

import numpy as np
import polars as pl

from src.models.predict import (
    _completed_weeks,
    _future_player_weeks,
    _prediction_output_frame,
    _remaining_weeks,
    _strict_preseason_snapshot,
)
from src.models.roles import STARTER_COUNTS, apply_starter_rule


def test_current_week_detection_excludes_unfinished_games() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 1,
                "game_type": "REG",
                "gameday": "2026-09-10",
                "gametime": "20:15",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 24,
                "away_score": 17,
            },
            {
                "season": 2026,
                "week": 2,
                "game_type": "REG",
                "gameday": "2026-09-17",
                "gametime": "20:15",
                "home_team": "BUF",
                "away_team": "NYJ",
                "home_score": None,
                "away_score": None,
            },
            {
                "season": 2026,
                "week": 3,
                "game_type": "REG",
                "gameday": "2026-09-24",
                "gametime": "20:15",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 28,
                "away_score": 24,
            },
            {
                "season": 2026,
                "week": 3,
                "game_type": "REG",
                "gameday": "2026-09-25",
                "gametime": "20:15",
                "home_team": "NYJ",
                "away_team": "DET",
                "home_score": None,
                "away_score": None,
            },
        ]
    )
    team_stats = pl.DataFrame(
        {
            "season": [2026, 2026, 2026, 2026],
            "week": [1, 1, 3, 3],
            "season_type": ["REG", "REG", "REG", "REG"],
        }
    )
    stats = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [1, 2, 3],
            "season_type": ["REG", "REG", "REG"],
        }
    )

    completed = _completed_weeks(stats, team_stats, schedules, 2026)
    remaining = _remaining_weeks(
        schedules,
        2026,
        datetime(2026, 9, 15, tzinfo=timezone.utc),
        completed,
    )

    assert completed == [1]
    assert remaining == [2]


def test_strict_preseason_snapshot_masks_scores_and_same_day_rankings() -> None:
    schedules = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": week,
                "game_type": "REG",
                "gameday": gameday,
                "gametime": gametime,
                "game_id": f"game-{week}",
                "home_team": "BUF",
                "away_team": "MIA",
                "home_score": 24,
                "away_score": 17,
            }
            for week, gameday, gametime in (
                (1, "2026-09-09", "20:20"),
                (2, "2026-09-17", "20:15"),
            )
        ]
    )
    rankings = pl.DataFrame(
        {
            "scrape_date": ["2026-09-04", "2026-09-09", "2026-09-10"],
            "ecr": [10.0, 5.0, 1.0],
        }
    )

    (
        preseason_schedule,
        preseason_rankings,
        first_kickoff,
        ranking_cutoff,
    ) = (
        _strict_preseason_snapshot(schedules, rankings, season=2026)
    )

    assert first_kickoff.isoformat() == "2026-09-10T00:20:00+00:00"
    assert ranking_cutoff == "2026-09-09"
    assert preseason_schedule["home_score"].null_count() == schedules.height
    assert preseason_schedule["away_score"].null_count() == schedules.height
    assert preseason_rankings["scrape_date"].to_list() == ["2026-09-04"]


def test_future_roster_rows_include_matchups_byes_and_defenses() -> None:
    rosters = pl.DataFrame(
        [
            {
                "week": 3,
                "game_type": "REG",
                "status": "ACT",
                "gsis_id": "qb-1",
                "position": "QB",
                "team": "BUF",
                "full_name": "Quarterback One",
                "football_name": None,
            },
            {
                "week": 3,
                "game_type": "REG",
                "status": "RES",
                "gsis_id": "fb-1",
                "position": "FB",
                "team": "NYJ",
                "full_name": "Fullback One",
                "football_name": None,
            },
            {
                "week": 3,
                "game_type": "REG",
                "status": "RLS",
                "gsis_id": "cut-1",
                "position": "RB",
                "team": "BUF",
                "full_name": "Released Player",
                "football_name": None,
            },
            {
                "week": 3,
                "game_type": "REG",
                "status": "ACT",
                "gsis_id": "ol-1",
                "position": "T",
                "team": "BUF",
                "full_name": "Offensive Lineman",
                "football_name": None,
            },
        ]
    )
    schedules = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 3,
                "game_type": "REG",
                "home_team": "BUF",
                "away_team": "MIA",
            },
            {
                "season": 2026,
                "week": 4,
                "game_type": "REG",
                "home_team": "NYJ",
                "away_team": "DET",
            },
        ]
    )
    history_schema = pl.Schema(
        {
            "season": pl.Int64,
            "week": pl.Int64,
            "season_type": pl.String,
            "player_id": pl.String,
            "player_name": pl.String,
            "player_display_name": pl.String,
            "source_position": pl.String,
            "position": pl.String,
            "team": pl.String,
            "opponent_team": pl.String,
            "full_ppr_points": pl.Float64,
            "stats_observed": pl.Boolean,
            "score_source": pl.String,
            "is_bye_week": pl.Boolean,
            "is_cancelled_game_week": pl.Boolean,
            "roster_status": pl.String,
            "points_allowed": pl.Int64,
        }
    )

    future = _future_player_weeks(
        rosters,
        schedules,
        history_schema,
        season=2026,
        week=3,
        roster_week=3,
    )
    players = {
        row["player_id"]: row for row in future.filter(
            pl.col("position") != "DST"
        ).to_dicts()
    }

    assert players["qb-1"]["opponent_team"] == "MIA"
    assert players["qb-1"]["is_bye_week"] is False
    assert players["fb-1"]["position"] == "RB"
    assert players["fb-1"]["is_bye_week"] is True
    assert "cut-1" not in players
    assert "ol-1" not in players
    assert future.filter(pl.col("position") == "DST").height == 4
    assert future.filter(
        (pl.col("player_id") == "DST:NYJ") & pl.col("is_bye_week")
    ).height == 1


def test_predictions_are_rejoined_with_qb_floor_and_bye_zeros() -> None:
    future_rows = pl.DataFrame(
        [
            {
                "season": 2026,
                "week": 12,
                "player_id": "player-2",
                "player_name": "Bye Player",
                "position": "RB",
                "team": "NYJ",
                "opponent_team": None,
                "roster_status": "ACT",
                "is_bye_week": True,
            },
            {
                "season": 2026,
                "week": 12,
                "player_id": "player-1",
                "player_name": "Playing Player",
                "position": "QB",
                "team": "BUF",
                "opponent_team": "MIA",
                "roster_status": "ACT",
                "is_bye_week": False,
            },
            {
                "season": 2026,
                "week": 12,
                "player_id": "player-3",
                "player_name": "Playing Running Back",
                "position": "RB",
                "team": "BUF",
                "opponent_team": "MIA",
                "roster_status": "ACT",
                "is_bye_week": False,
            },
        ]
    )
    forecast_features = pl.DataFrame(
        {
            "season": [2026, 2026, 2026],
            "week": [12, 12, 12],
            "player_id": ["player-1", "player-3", "player-2"],
            "position": ["QB", "RB", "RB"],
            "feature_is_bye_week": [False, False, True],
        }
    )

    output = _prediction_output_frame(
        future_rows,
        forecast_features,
        np.array([-3.5, -1.25, 0.0]),
    )
    scores = dict(
        output.select("player_id", "predicted_full_ppr_points").iter_rows()
    )

    assert scores == {"player-1": 0.0, "player-2": 0.0, "player-3": -1.25}


def test_starter_slots_are_selected_by_team_position_rank() -> None:
    rows = []
    for position, starter_count in STARTER_COUNTS.items():
        for rank in range(1, starter_count + 2):
            rows.append(
                {
                    "season": 2026,
                    "week": 1,
                    "team": "BUF",
                    "position": position,
                    "player_id": f"{position}-{rank}",
                    "feature_fantasypros_position_rank": rank,
                    "feature_is_bye_week": False,
                }
            )
    features = pl.DataFrame(rows)
    scores = np.full(features.height, 10.0)
    scores[0] = -3.0

    weighted, conditional, starters, multipliers = apply_starter_rule(
        features,
        scores,
    )
    starter_counts = {
        position: sum(
            is_starter
            for row, is_starter in zip(rows, starters, strict=True)
            if row["position"] == position
        )
        for position in STARTER_COUNTS
    }

    assert starter_counts == STARTER_COUNTS
    assert conditional[0] == 0.0
    assert weighted.tolist() == (conditional * multipliers).tolist()
    assert set(multipliers.tolist()) == {0.5, 1.0}


def test_unranked_players_fill_open_slots_by_conditional_score() -> None:
    features = pl.DataFrame(
        {
            "season": [2026] * 4,
            "week": [1] * 4,
            "team": ["BUF"] * 4,
            "position": ["RB"] * 4,
            "player_id": ["ranked", "unranked-low", "unranked-high", "unranked-mid"],
            "feature_fantasypros_position_rank": [1.0, None, None, None],
            "feature_is_bye_week": [False] * 4,
        }
    )

    _, _, starters, _ = apply_starter_rule(
        features,
        np.array([10.0, 5.0, 20.0, 15.0]),
    )

    assert starters.tolist() == [True, False, True, True]
