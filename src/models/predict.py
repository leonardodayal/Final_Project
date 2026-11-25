"""Generate live or strict-preseason weekly player-score projections."""

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.data.clean import ROSTER_STATUSES, build_player_week_dataset
from src.features.build import FEATURE_COLUMNS, build_feature_dataset
from src.models.roles import apply_starter_rule

EASTERN = ZoneInfo("America/New_York")
FANTASY_POSITIONS = {"QB", "RB", "FB", "WR", "TE", "K"}


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "The as-of timestamp must be ISO formatted."
        ) from error
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError(
            "The as-of timestamp must include a timezone."
        )
    return parsed.astimezone(timezone.utc)


def _kickoff_utc(gameday: str, gametime: str) -> datetime:
    local_kickoff = datetime.strptime(
        f"{gameday} {gametime}",
        "%Y-%m-%d %H:%M",
    ).replace(tzinfo=EASTERN)
    return local_kickoff.astimezone(timezone.utc)


def _completed_weeks(
    stats: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    season: int,
) -> list[int]:
    regular_schedule = schedules.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    schedule_weeks = regular_schedule.group_by("week").agg(
        pl.len().alias("_scheduled_games"),
        (
            pl.col("home_score").is_not_null()
            & pl.col("away_score").is_not_null()
        )
        .sum()
        .alias("_final_games"),
    )
    team_weeks = (
        team_stats.filter(
            (pl.col("season") == season) & (pl.col("season_type") == "REG")
        )
        .group_by("week")
        .agg(pl.len().alias("_team_rows"))
    )
    player_weeks = (
        stats.filter(
            (pl.col("season") == season) & (pl.col("season_type") == "REG")
        )
        .group_by("week")
        .agg(pl.len().alias("_player_rows"))
    )
    completed = (
        schedule_weeks.join(team_weeks, on="week", how="inner")
        .join(player_weeks, on="week", how="inner")
        .filter(
            (pl.col("_scheduled_games") == pl.col("_final_games"))
            & (pl.col("_team_rows") == 2 * pl.col("_scheduled_games"))
        )
        .get_column("week")
        .to_list()
    )
    return sorted(int(week) for week in completed)


def _remaining_weeks(
    schedules: pl.DataFrame,
    season: int,
    as_of: datetime,
    completed_weeks: list[int],
) -> list[int]:
    regular_schedule = schedules.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    weeks: dict[int, dict] = {}
    for game in regular_schedule.select(
        "week",
        "gameday",
        "gametime",
        "home_score",
        "away_score",
    ).to_dicts():
        if game["gameday"] is None or game["gametime"] is None:
            raise ValueError(
                "The regular-season schedule has a missing kickoff time."
            )
        week = int(game["week"])
        kickoff = _kickoff_utc(game["gameday"], game["gametime"])
        values = weeks.setdefault(
            week,
            {"kickoffs": [], "has_final_score": False},
        )
        values["kickoffs"].append(kickoff)
        values["has_final_score"] = values["has_final_score"] or (
            game["home_score"] is not None or game["away_score"] is not None
        )
    return [
        week
        for week, values in sorted(weeks.items())
        if week not in completed_weeks
        and min(values["kickoffs"]) > as_of
        and not values["has_final_score"]
    ]


def _strict_preseason_snapshot(
    schedules: pl.DataFrame,
    rankings: pl.DataFrame,
    season: int,
) -> tuple[pl.DataFrame, pl.DataFrame, datetime, str]:
    regular_schedule = schedules.filter(
        (pl.col("season") == season) & (pl.col("game_type") == "REG")
    )
    kickoff_rows = regular_schedule.select("gameday", "gametime").to_dicts()
    if not kickoff_rows:
        raise ValueError(f"No regular-season schedule found for {season}.")
    kickoffs = []
    for game in kickoff_rows:
        if game["gameday"] is None or game["gametime"] is None:
            raise ValueError(
                "The regular-season schedule has a missing kickoff time."
            )
        kickoffs.append(_kickoff_utc(game["gameday"], game["gametime"]))
    first_kickoff = min(kickoffs)
    cutoff_date = min(game["gameday"] for game in kickoff_rows)
    preseason_rankings = rankings.filter(
        pl.col("scrape_date").cast(pl.String) < cutoff_date
    )
    preseason_schedules = schedules.with_columns(
        pl.when(pl.col("season") == season)
        .then(pl.lit(None).cast(schedules.schema["home_score"]))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(pl.col("season") == season)
        .then(pl.lit(None).cast(schedules.schema["away_score"]))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )
    return preseason_schedules, preseason_rankings, first_kickoff, cutoff_date


def _empty_team_stats(schedules: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": schedules.schema["season"],
            "week": schedules.schema["week"],
            "season_type": pl.String,
            "game_id": schedules.schema["game_id"],
            "team": pl.String,
            "opponent_team": pl.String,
            "passing_yards": pl.Float64,
            "rushing_yards": pl.Float64,
            "def_tds": pl.Int64,
        }
    )


def _future_player_weeks(
    roster_snapshot: pl.DataFrame,
    schedules: pl.DataFrame,
    history_schema: pl.Schema,
    season: int,
    week: int,
    roster_week: int,
) -> pl.DataFrame:
    required_roster_columns = {
        "week",
        "game_type",
        "status",
        "gsis_id",
        "position",
        "team",
        "full_name",
    }
    missing = required_roster_columns - set(roster_snapshot.columns)
    if missing:
        raise ValueError(
            "Weekly rosters are missing required columns: "
            + ", ".join(sorted(missing))
        )

    games = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
    )
    opponents: dict[str, str] = {}
    for game in games.select("home_team", "away_team").to_dicts():
        for team, opponent in (
            (game["home_team"], game["away_team"]),
            (game["away_team"], game["home_team"]),
        ):
            if team in opponents:
                raise ValueError(
                    f"Team {team} has more than one scheduled game in week {week}."
                )
            opponents[team] = opponent

    active_roster = roster_snapshot.filter(
        (pl.col("week") == roster_week)
        & (pl.col("game_type") == "REG")
        & pl.col("status").is_in(list(ROSTER_STATUSES))
        & pl.col("position").is_in(list(FANTASY_POSITIONS))
        & pl.col("gsis_id").is_not_null()
        & pl.col("team").is_not_null()
    )
    if active_roster.select(pl.col("gsis_id").is_duplicated().any()).item():
        raise ValueError(
            f"Weekly rosters contain duplicate player IDs in week {roster_week}."
        )

    records = []

    def new_record() -> dict:
        return {column: None for column in history_schema}

    for player in active_roster.to_dicts():
        team = player["team"]
        position = "RB" if player["position"] == "FB" else player["position"]
        record = new_record()
        record.update(
            {
                "season": season,
                "week": week,
                "season_type": "REG",
                "player_id": str(player["gsis_id"]),
                "player_name": player["full_name"],
                "player_display_name": (
                    player.get("football_name")
                    or player["full_name"]
                ),
                "source_position": player["position"],
                "position": position,
                "team": team,
                "opponent_team": opponents.get(team),
                "full_ppr_points": None,
                "stats_observed": False,
                "score_source": "forecast_placeholder",
                "is_bye_week": team not in opponents,
                "is_cancelled_game_week": False,
                "roster_status": player["status"],
                "points_allowed": None,
            }
        )
        records.append(record)

    teams = set(schedules.filter(pl.col("season") == season)["home_team"].to_list())
    teams.update(
        schedules.filter(pl.col("season") == season)["away_team"].to_list()
    )
    for team in sorted(team for team in teams if team is not None):
        record = new_record()
        is_bye = team not in opponents
        record.update(
            {
                "season": season,
                "week": week,
                "season_type": "REG",
                "player_id": f"DST:{team}",
                "player_name": f"{team} D/ST",
                "player_display_name": f"{team} D/ST",
                "source_position": "DST",
                "position": "DST",
                "team": team,
                "opponent_team": opponents.get(team),
                "full_ppr_points": None,
                "stats_observed": False,
                "score_source": (
                    "forecast_placeholder" if is_bye else "forecast_context"
                ),
                "is_bye_week": is_bye,
                "is_cancelled_game_week": False,
                "roster_status": None,
                "points_allowed": None if is_bye else 0,
            }
        )
        records.append(record)

    if not records:
        raise ValueError(f"No forecast candidates were built for week {week}.")
    return pl.DataFrame(records, schema=history_schema, strict=False)


def _future_team_stats(
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    games = schedules.filter(
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
    )
    records = []
    for game in games.select(
        "game_id",
        "home_team",
        "away_team",
    ).to_dicts():
        for team, opponent in (
            (game["home_team"], game["away_team"]),
            (game["away_team"], game["home_team"]),
        ):
            record = {column: None for column in team_stats.columns}
            record.update(
                {
                    "season": season,
                    "week": week,
                    "season_type": "REG",
                    "game_id": game["game_id"],
                    "team": team,
                    "opponent_team": opponent,
                    "passing_yards": 0.0,
                    "rushing_yards": 0.0,
                    "def_tds": 0,
                }
            )
            records.append(record)
    if not records:
        raise ValueError(f"No scheduled games were found for week {week}.")
    # These synthetic rows let the existing shifted-history logic emit
    # target-week matchup features without using future game results.
    placeholder_stats = pl.DataFrame(
        records,
        schema=team_stats.schema,
        strict=False,
    )
    return pl.concat([team_stats, placeholder_stats], how="diagonal_relaxed")


def _schedule_with_future_scores(
    schedules: pl.DataFrame,
    season: int,
    week: int,
) -> pl.DataFrame:
    target_week = (
        (pl.col("season") == season)
        & (pl.col("week") == week)
        & (pl.col("game_type") == "REG")
    )
    return schedules.with_columns(
        pl.when(target_week)
        .then(pl.lit(0).cast(schedules.schema["home_score"]))
        .otherwise(pl.col("home_score"))
        .alias("home_score"),
        pl.when(target_week)
        .then(pl.lit(0).cast(schedules.schema["away_score"]))
        .otherwise(pl.col("away_score"))
        .alias("away_score"),
    )


def _prediction_output_frame(
    future_rows: pl.DataFrame,
    forecast_features: pl.DataFrame,
    predictions: np.ndarray,
    conditional_scores: np.ndarray | None = None,
    predicted_starter: np.ndarray | None = None,
    starter_multiplier: np.ndarray | None = None,
) -> pl.DataFrame:
    quarterback_rows = forecast_features["position"].to_numpy() == "QB"
    predictions = np.where(
        quarterback_rows,
        np.maximum(predictions, 0.0),
        predictions,
    )
    prediction_columns = [
        pl.Series("predicted_full_ppr_points", predictions)
    ]
    if conditional_scores is not None:
        prediction_columns.append(
            pl.Series("predicted_full_ppr_points_if_active", conditional_scores)
        )
    if predicted_starter is not None:
        prediction_columns.append(
            pl.Series(
                "predicted_starter",
                predicted_starter,
            )
        )
    if starter_multiplier is not None:
        prediction_columns.append(
            pl.Series(
                "starter_score_multiplier",
                starter_multiplier,
            )
        )
    keyed_predictions = forecast_features.select(
        "season",
        "week",
        "player_id",
        pl.col("feature_is_bye_week").alias("_model_is_bye_week"),
    ).with_columns(prediction_columns)
    result = future_rows.select(
        "season",
        "week",
        "player_id",
        "player_name",
        "position",
        "team",
        "opponent_team",
        "roster_status",
        "is_bye_week",
    ).join(
        keyed_predictions,
        on=["season", "week", "player_id"],
        how="left",
        validate="1:1",
    )
    if result["predicted_full_ppr_points"].null_count():
        raise ValueError("A forecast candidate did not receive a prediction.")
    if result.filter(
        pl.col("is_bye_week") != pl.col("_model_is_bye_week")
    ).height:
        raise ValueError("Forecast and feature bye-week flags do not match.")
    if result.filter(
        pl.col("is_bye_week")
        & (pl.col("predicted_full_ppr_points") != 0)
    ).height:
        raise ValueError("Bye-week predictions must be exactly zero.")
    return result.drop("_model_is_bye_week")


def generate_rest_of_season_predictions(
    season: int,
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
    as_of: datetime | None = None,
    strict_preseason: bool = False,
) -> dict:
    """Generate weekly projections using only data available at the cutoff."""
    if strict_preseason and as_of is not None:
        raise ValueError("--as-of cannot be combined with strict preseason mode.")
    if strict_preseason and season != 2026:
        raise ValueError("Strict preseason mode is currently configured for 2026.")
    if as_of is None and not strict_preseason:
        as_of = datetime.now(timezone.utc)
    elif as_of is not None and as_of.tzinfo is None:
        raise ValueError("The as-of timestamp must include a timezone.")
    elif as_of is not None:
        as_of = as_of.astimezone(timezone.utc)
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    stats_path = raw_dir / f"weekly_player_stats_{season}.parquet"
    historical_path = (
        processed_dir
        / "player_weekly_2021-2022-2023-2024-2025.parquet"
    )
    ranking_path = raw_dir / "historical_fantasy_rankings.parquet"
    player_ids_path = raw_dir / "fantasy_player_id_crosswalk.parquet"
    required_paths = [historical_path, ranking_path, player_ids_path]
    if not strict_preseason:
        required_paths.append(stats_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}. Collect and clean the "
                "season data before forecasting."
            )

    import nflreadpy as nfl

    rosters = nfl.load_rosters_weekly(seasons=[season])
    schedules = nfl.load_schedules(seasons=[season])
    rosters.write_parquet(raw_dir / f"weekly_rosters_{season}.parquet")
    schedules.write_parquet(raw_dir / f"schedules_{season}.parquet")
    player_weeks = pl.read_parquet(historical_path)
    rankings = pl.read_parquet(ranking_path).with_columns(
        pl.col("scrape_date").cast(pl.String)
    )
    player_ids = pl.read_parquet(player_ids_path)
    injury_feature_data = None

    if strict_preseason:
        (
            schedules,
            rankings,
            first_kickoff,
            as_of_date,
        ) = _strict_preseason_snapshot(
            schedules,
            rankings,
            season,
        )
        forecast_weeks = sorted(
            int(week)
            for week in schedules.filter(
                (pl.col("season") == season) & (pl.col("game_type") == "REG")
            )["week"].unique().to_list()
        )
        if forecast_weeks != list(range(1, 19)):
            raise ValueError(
                "Strict preseason mode requires a complete regular-season "
                "schedule for Weeks 1–18."
            )
        if rosters.filter(
            (pl.col("week") == 1) & (pl.col("game_type") == "REG")
        ).is_empty():
            raise ValueError("No Week 1 roster snapshot was returned.")
        roster_week = 1
        last_completed_week = None
        completed_team_stats = _empty_team_stats(schedules)
        as_of = first_kickoff - timedelta(microseconds=1)
        injury_note = (
            "Excluded in strict preseason mode; no in-season injury reports "
            "are used."
        )
    else:
        stats = pl.read_parquet(stats_path)
        team_stats = nfl.load_team_stats(
            seasons=[season],
            summary_level="week",
        )
        injuries = nfl.load_injuries(seasons=[season])
        team_stats.write_parquet(raw_dir / f"weekly_team_stats_{season}.parquet")
        injuries.write_parquet(raw_dir / f"weekly_injuries_{season}.parquet")

        completed_weeks = _completed_weeks(
            stats,
            team_stats,
            schedules,
            season,
        )
        if not completed_weeks:
            raise ValueError(
                f"No completed regular-season games found for {season}."
            )
        forecast_weeks = _remaining_weeks(
            schedules,
            season,
            as_of,
            completed_weeks,
        )
        if not forecast_weeks:
            raise ValueError(
                f"No fully unplayed regular-season weeks remain after "
                f"{as_of.isoformat()}."
            )
        last_completed_week = max(completed_weeks)
        roster_weeks = rosters.filter(
            (pl.col("game_type") == "REG")
            & (pl.col("week") <= forecast_weeks[0])
        )["week"]
        if roster_weeks.is_empty():
            raise ValueError("No eligible weekly roster snapshot was returned.")
        roster_week = int(roster_weeks.max())

        completed_stats = stats.filter(
            (pl.col("season") == season)
            & (pl.col("season_type") == "REG")
            & pl.col("week").is_in(completed_weeks)
        )
        completed_rosters = rosters.filter(
            (pl.col("season") == season)
            & (pl.col("game_type") == "REG")
            & pl.col("week").is_in(completed_weeks)
        )
        completed_team_stats = team_stats.filter(
            (pl.col("season") == season)
            & (pl.col("season_type") == "REG")
            & pl.col("week").is_in(completed_weeks)
        )
        completed_schedules = schedules.filter(
            (pl.col("season") == season)
            & (pl.col("game_type") == "REG")
            & pl.col("week").is_in(completed_weeks)
        )
        current_history = build_player_week_dataset(
            completed_stats,
            completed_rosters,
            completed_team_stats,
            completed_schedules,
        )
        processed_dir.mkdir(parents=True, exist_ok=True)
        current_history.write_parquet(
            processed_dir
            / f"player_weekly_{season}_completed_through_{last_completed_week}.parquet"
        )
        player_weeks = pl.concat(
            [player_weeks, current_history],
            how="diagonal_relaxed",
        )

        as_of_date = as_of.date().isoformat()
        rankings = rankings.filter(pl.col("scrape_date") <= as_of_date)
        injury_note = (
            "Unknown: the current injury feed has no date_modified timestamp, "
            "so report status cannot be verified as-of the forecast."
        )
        if "date_modified" in injuries.columns:
            timestamp_type = injuries.schema["date_modified"]
            if (
                timestamp_type.base_type() == pl.Datetime
                and timestamp_type.time_zone == "UTC"
            ):
                injury_feature_data = injuries.filter(
                    pl.col("date_modified").is_not_null()
                    & (pl.col("date_modified") <= pl.lit(as_of))
                )
                injury_note = (
                    "Latest timestamped report strictly before each target "
                    "week's first kickoff, capped at the forecast as-of time."
                )
            else:
                injury_note = (
                    "Unknown: injury timestamps are present but are not "
                    "explicitly UTC, so they are excluded to avoid leakage."
                )

    feature_schedules = schedules

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or not {
        "model",
        "feature_columns",
        "selected_model",
        "conditional_score_adjustments",
        "starter_counts",
        "backup_score_multiplier",
    }.issubset(artifact):
        raise ValueError(f"Model artifact has an unsupported format: {model_path}")
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
    conditional_score_adjustments = artifact["conditional_score_adjustments"]
    starter_counts = artifact["starter_counts"]
    backup_score_multiplier = artifact["backup_score_multiplier"]
    missing_features = set(feature_columns) - set(FEATURE_COLUMNS)
    if missing_features:
        raise ValueError(
            "Model artifact expects unknown features: "
            + ", ".join(sorted(missing_features))
        )
    if not isinstance(conditional_score_adjustments, dict):
        raise ValueError("Model artifact has invalid score-calibration data.")
    if not isinstance(starter_counts, dict) or not isinstance(
        backup_score_multiplier,
        (int, float),
    ):
        raise ValueError("Model artifact has invalid starter-scoring rules.")

    weekly_predictions = []
    for week in forecast_weeks:
        future_rows = _future_player_weeks(
            rosters,
            feature_schedules,
            player_weeks.schema,
            season,
            week,
            roster_week,
        )
        feature_rows = pl.concat(
            [player_weeks, future_rows],
            how="diagonal_relaxed",
        )
        forecast_features = build_feature_dataset(
            feature_rows,
            _future_team_stats(
                completed_team_stats,
                feature_schedules,
                season,
                week,
            ),
            _schedule_with_future_scores(feature_schedules, season, week),
            rankings,
            player_ids,
            injuries=injury_feature_data,
            output_seasons=[season],
        ).filter(pl.col("week") == week)
        if forecast_features.height != future_rows.height:
            raise ValueError(
                f"Feature generation changed the number of week {week} "
                "forecast candidates."
            )
        feature_frame = pd.DataFrame(
            {
                column: forecast_features.get_column(column).to_list()
                for column in feature_columns
            }
        )
        non_bye = ~forecast_features["feature_is_bye_week"].to_numpy()
        raw_scores = np.zeros(forecast_features.height, dtype=float)
        if non_bye.any():
            raw_scores[non_bye] = model.predict(
                feature_frame.loc[non_bye]
            )
        positions = forecast_features["position"].to_list()
        calibrated_scores = raw_scores + np.array(
            [
                conditional_score_adjustments.get(position, 0.0)
                for position in positions
            ],
            dtype=float,
        )
        (
            predictions,
            conditional_scores,
            predicted_starter,
            starter_multiplier,
        ) = apply_starter_rule(
            forecast_features,
            calibrated_scores,
            starter_counts,
            backup_score_multiplier,
        )
        weekly_predictions.append(
            _prediction_output_frame(
                future_rows,
                forecast_features,
                predictions,
                conditional_scores,
                predicted_starter,
                starter_multiplier,
            )
        )

    predictions = pl.concat(weekly_predictions, how="vertical_relaxed").sort(
        ["week", "position", "player_id"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = "_preseason" if strict_preseason else ""
    prediction_path = output_dir / (
        f"weekly_predictions_{season}{output_suffix}.parquet"
    )
    predictions.write_parquet(prediction_path)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "forecast_mode": (
            "strict_preseason_snapshot" if strict_preseason else "live"
        ),
        "season": season,
        "last_completed_week": last_completed_week,
        "roster_snapshot_week": roster_week,
        "forecast_weeks": forecast_weeks,
        "candidate_rows": predictions.height,
        "model_artifact": str(model_path),
        "selected_model": artifact["selected_model"],
        "model_training_seasons": [2021, 2022, 2023, 2024],
        "history_seasons": [2021, 2022, 2023, 2024, 2025],
        "ranking_snapshots_cut_off_at": as_of_date,
        "ranking_cutoff_is_exclusive": strict_preseason,
        "injury_features": injury_note,
        "bye_week_policy": "Predictions are forced to exactly zero.",
        "quarterback_floor_policy": (
            "Conditional weekly QB scores are floored at zero; non-QB scores "
            "are unchanged."
        ),
        "starter_scoring_policy": (
            "Within each team and position, the highest FantasyPros-ranked "
            "players receive full if-active scores: 1 QB, 3 RB, 4 WR, 2 TE, "
            "1 K, and 1 DST. Other players receive half. Unranked players fill "
            "open starter slots by conditional score."
        ),
        "multiweek_feature_policy": (
            "All 2026 player and team results are excluded; historical player "
            "data ends in 2025, and 2026 matchup-history features remain "
            "unknown."
            if strict_preseason
            else (
                "Every future week uses player and team history through the "
                "last completed week; future scores are not recursively "
                "invented."
            )
        ),
    }
    metadata_path = output_dir / (
        f"weekly_predictions_{season}{output_suffix}.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"forecast_weeks: {forecast_weeks[0]}-{forecast_weeks[-1]} "
        f"({len(forecast_weeks)} weeks)"
    )
    print(f"candidate_rows: {predictions.height:,}")
    print(f"predictions: {prediction_path}")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="Current NFL season, for example 2026",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing raw and processed data",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/weekly_score_model.joblib"),
        help="Serialized Step 4 model artifact",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for weekly forecast Parquet and metadata",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_as_of,
        default=None,
        help="Timezone-aware ISO timestamp; defaults to the current UTC time",
    )
    parser.add_argument(
        "--strict-preseason",
        action="store_true",
        help=(
            "Forecast Weeks 1–18 using only Week 1 rosters, pre-opening "
            "rankings, and player history through 2025"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    generate_rest_of_season_predictions(
        args.season,
        args.data_dir,
        args.model_path,
        args.output_dir,
        args.as_of,
        args.strict_preseason,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
