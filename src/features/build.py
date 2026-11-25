"""Build leakage-safe features for weekly fantasy-score predictions."""

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

WINDOWS = (3, 5)
PLAYER_HISTORY_METRICS = {
    "full_ppr_points": "score",
    "carries": "carries",
    "targets": "targets",
    "receptions": "receptions",
    "attempts": "passing_attempts",
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
}
DEFENSE_HISTORY_METRICS = {
    "passing_yards_allowed": "passing_yards_allowed",
    "rushing_yards_allowed": "rushing_yards_allowed",
    "points_allowed": "points_allowed",
}
OFFENSE_HISTORY_METRICS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "points_scored": "points_scored",
}
WEEKLY_RANKING_PAGE_POSITIONS = {
    "weekly-qb": "QB",
    "weekly-rb": "RB",
    "weekly-wr": "WR",
    "weekly-te": "TE",
    "weekly-k": "K",
    "weekly-dst": "DST",
}
REDRAFT_RANKING_PAGE_POSITIONS = {
    "redraft-qb": "QB",
    "redraft-rb": "RB",
    "redraft-wr": "WR",
    "redraft-te": "TE",
    "redraft-k": "K",
    "redraft-dst": "DST",
}
RANKING_PAGE_POSITIONS = {
    **WEEKLY_RANKING_PAGE_POSITIONS,
    **REDRAFT_RANKING_PAGE_POSITIONS,
}
DST_TEAM_ALIASES = {"JAC": "JAX", "LAR": "LA"}
RANKING_FEATURE_COLUMNS = (
    "feature_fantasypros_position_rank",
    "feature_fantasypros_rank_source",
    "feature_fantasypros_rank_available",
)
INJURY_FEATURE_COLUMNS = (
    "feature_injury_report_status",
    "feature_injury_practice_status",
    "feature_injury_report_available",
)


def _require_columns(
    frame: pl.DataFrame,
    required: set[str],
    dataset: str,
) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{dataset} is missing required columns: {', '.join(sorted(missing))}"
        )


def _has_duplicate_keys(frame: pl.DataFrame, keys: list[str]) -> bool:
    return frame.select(pl.struct(keys).is_duplicated().any()).item()


def _week_cutoffs(schedules: pl.DataFrame) -> pl.DataFrame:
    _require_columns(
        schedules,
        {"season", "week", "game_type", "gameday", "gametime"},
        "Schedules",
    )
    regular = schedules.filter(pl.col("game_type") == "REG")
    kickoff = pl.concat_str(
        [pl.col("gameday"), pl.lit(" "), pl.col("gametime")]
    ).str.to_datetime(
        format="%Y-%m-%d %H:%M",
        time_zone="America/New_York",
        time_unit="us",
    )
    return (
        regular.with_columns(
            kickoff.dt.convert_time_zone("UTC").alias("_kickoff_utc")
        )
        .group_by(["season", "week"])
        .agg(
            pl.col("_kickoff_utc").min().alias("_cutoff_utc"),
            pl.col("gameday").min().alias("_cutoff_date"),
        )
    )


def _rank_features(
    player_weeks: pl.DataFrame,
    rankings: pl.DataFrame,
    player_ids: pl.DataFrame,
    cutoffs: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        rankings,
        {"page_type", "ecr_type", "id", "ecr", "scrape_date", "team"},
        "FantasyPros rankings",
    )
    _require_columns(
        player_ids,
        {"fantasypros_id", "gsis_id"},
        "FantasyPros player crosswalk",
    )
    mapping = (
        player_ids.select(
            pl.col("fantasypros_id").cast(pl.Int64, strict=False),
            pl.col("gsis_id").cast(pl.String).alias("player_id"),
        )
        .drop_nulls()
        .unique()
    )
    if _has_duplicate_keys(mapping, ["fantasypros_id"]):
        raise ValueError("FantasyPros player IDs map to multiple NFL players.")

    rank_snapshots = (
        rankings.filter(
            (
                pl.col("page_type").is_in(list(WEEKLY_RANKING_PAGE_POSITIONS))
                & (pl.col("ecr_type") == "wp")
            )
            | (
                pl.col("page_type").is_in(list(REDRAFT_RANKING_PAGE_POSITIONS))
                & (pl.col("ecr_type") == "rp")
            )
        )
        .with_columns(
            pl.col("ecr").cast(pl.Float64, strict=False),
            pl.col("scrape_date").cast(pl.String),
            pl.col("page_type")
            .replace(RANKING_PAGE_POSITIONS)
            .alias("position"),
            pl.when(
                pl.col("page_type").is_in(list(WEEKLY_RANKING_PAGE_POSITIONS))
            )
            .then(pl.lit("weekly"))
            .otherwise(pl.lit("redraft"))
            .alias("_rank_source"),
        )
        .with_columns(
            pl.col("scrape_date").str.slice(0, 4).cast(pl.Int32).alias("season")
        )
    )
    skill_rankings = (
        rank_snapshots.filter(pl.col("position") != "DST")
        .with_columns(
            pl.col("id").cast(pl.Int64, strict=False).alias("_fantasypros_id")
        )
        .join(
            mapping,
            left_on="_fantasypros_id",
            right_on="fantasypros_id",
            how="inner",
        )
        .select(
            "season",
            "player_id",
            "position",
            "scrape_date",
            "ecr",
            "_rank_source",
        )
    )
    defense_rankings = (
        rank_snapshots.filter(
            (pl.col("position") == "DST") & pl.col("team").is_not_null()
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("DST:"),
                    pl.col("team")
                    .cast(pl.String)
                    .str.to_uppercase()
                    .replace(DST_TEAM_ALIASES),
                ]
            ).alias("player_id")
        )
        .select(
            "season",
            "player_id",
            "position",
            "scrape_date",
            "ecr",
            "_rank_source",
        )
    )
    rank_history = pl.concat(
        [skill_rankings, defense_rankings],
        how="vertical_relaxed",
    ).filter(pl.col("ecr").is_not_null()).with_columns(
        pl.when(pl.col("_rank_source") == "weekly")
        .then(pl.lit(1))
        .otherwise(pl.lit(0))
        .alias("_rank_priority")
    )
    if _has_duplicate_keys(
        rank_history,
        ["season", "player_id", "position", "scrape_date", "_rank_source"],
    ):
        raise ValueError("FantasyPros rankings contain duplicate player snapshots.")

    target_keys = (
        player_weeks.select("season", "week", "player_id", "position")
        .unique()
        .join(cutoffs, on=["season", "week"], how="left")
    )
    eligible = (
        target_keys.join(
            rank_history,
            on=["season", "player_id", "position"],
            how="left",
        )
        .filter(
            pl.col("scrape_date").is_not_null()
            & (pl.col("scrape_date") < pl.col("_cutoff_date"))
        )
        .sort(
            [
                "season",
                "week",
                "player_id",
                "scrape_date",
                "_rank_priority",
            ]
        )
        .group_by(["season", "week", "player_id"])
        .agg(
            pl.col("ecr").last().alias(RANKING_FEATURE_COLUMNS[0]),
            pl.col("_rank_source").last().alias(RANKING_FEATURE_COLUMNS[1]),
        )
    )
    return (
        target_keys.select("season", "week", "player_id")
        .join(eligible, on=["season", "week", "player_id"], how="left")
        .with_columns(
            pl.col(RANKING_FEATURE_COLUMNS[0])
            .is_not_null()
            .alias(RANKING_FEATURE_COLUMNS[2])
        )
    )


def _injury_features(
    injuries: pl.DataFrame,
    cutoffs: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        injuries,
        {
            "season",
            "week",
            "game_type",
            "gsis_id",
            "date_modified",
            "report_status",
            "practice_status",
        },
        "Weekly injury reports",
    )
    clean_injuries = injuries.with_columns(
        pl.col("report_status")
        .cast(pl.String)
        .str.strip_chars()
        .replace("", None),
        pl.col("practice_status")
        .cast(pl.String)
        .str.strip_chars()
        .replace("", None),
        pl.col("gsis_id").cast(pl.String).alias("player_id"),
    )
    return (
        clean_injuries.filter(
            (pl.col("game_type") == "REG")
            & pl.col("player_id").is_not_null()
            & pl.col("date_modified").is_not_null()
        )
        .join(cutoffs, on=["season", "week"], how="inner")
        .filter(pl.col("date_modified") < pl.col("_cutoff_utc"))
        .sort(["season", "week", "player_id", "date_modified"])
        .group_by(["season", "week", "player_id"])
        .agg(
            pl.col("report_status")
            .drop_nulls()
            .last()
            .alias(INJURY_FEATURE_COLUMNS[0]),
            pl.col("practice_status")
            .drop_nulls()
            .last()
            .alias(INJURY_FEATURE_COLUMNS[1]),
            pl.lit(True).alias(INJURY_FEATURE_COLUMNS[2]),
        )
    )


def _rolling_features(
    frame: pl.DataFrame,
    entity_column: str,
    metrics: dict[str, str],
    prefix: str,
) -> tuple[pl.DataFrame, list[str]]:
    group_columns = ["season", entity_column]
    sorted_frame = frame.sort([*group_columns, "week"])
    expressions = []
    feature_columns = []
    for source_column, feature_name in metrics.items():
        for window in WINDOWS:
            column_name = (
                f"feature_{prefix}_{feature_name}_avg_last_{window}_games"
            )
            expressions.append(
                pl.col(source_column)
                .cast(pl.Float64)
                .fill_null(0)
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over(group_columns)
                .alias(column_name)
            )
            feature_columns.append(column_name)
    return sorted_frame.with_columns(expressions), feature_columns


def _player_history_features(
    player_weeks: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]]:
    player_group = ["player_id"]
    season_group = ["season", "player_id"]
    sorted_weeks = player_weeks.sort(["player_id", "season", "week"])
    values = {}
    expressions = []
    feature_columns = [
        "feature_prior_player_weeks",
        "feature_prior_stat_weeks",
    ]

    expressions.extend(
        [
            (
                pl.col("player_id")
                .cum_count()
                .shift(1)
                .over(player_group)
                .fill_null(0)
                .alias("feature_prior_player_weeks")
            ),
            (
                pl.col("stats_observed")
                .cast(pl.UInt32)
                .cum_sum()
                .shift(1)
                .over(player_group)
                .fill_null(0)
                .alias("feature_prior_stat_weeks")
            ),
        ]
    )

    for source_column, feature_name in PLAYER_HISTORY_METRICS.items():
        value_column = f"_history_{feature_name}"
        observed_stat = pl.col("stats_observed")
        if source_column == "full_ppr_points":
            value = (
                pl.when(observed_stat)
                .then(pl.col(source_column).cast(pl.Float64))
                .otherwise(None)
            )
        else:
            value = (
                pl.when((pl.col("position") != "DST") & observed_stat)
                .then(pl.col(source_column).cast(pl.Float64).fill_null(0))
                .otherwise(None)
            )
        values[feature_name] = value_column
        expressions.append(value.alias(value_column))

    history = sorted_weeks.with_columns(expressions)
    history_expressions = []
    for feature_name, value_column in values.items():
        lag_name = f"feature_previous_week_{feature_name}"
        history_expressions.append(
            pl.col(value_column).shift(1).over(player_group).alias(lag_name)
        )
        feature_columns.append(lag_name)

        for window in WINDOWS:
            rolling_name = f"feature_{feature_name}_avg_last_{window}_weeks"
            history_expressions.append(
                pl.col(value_column)
                .shift(1)
                .rolling_mean(window_size=window, min_samples=1)
                .over(player_group)
                .alias(rolling_name)
            )
            feature_columns.append(rolling_name)

        prior_sum = (
            pl.col(value_column)
            .fill_null(0)
            .cum_sum()
            .shift(1)
            .over(season_group)
        )
        prior_count = (
            pl.col(value_column)
            .is_not_null()
            .cast(pl.UInt32)
            .cum_sum()
            .shift(1)
            .over(season_group)
        )
        season_average = (
            pl.when(prior_count > 0)
            .then(prior_sum / prior_count)
            .otherwise(None)
        )
        season_average_name = f"feature_{feature_name}_avg_season_to_date"
        history_expressions.append(season_average.alias(season_average_name))
        feature_columns.append(season_average_name)

    history = history.with_columns(history_expressions).drop(list(values.values()))
    return history, feature_columns


def _schedule_context(schedules: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    _require_columns(
        schedules,
        {
            "season",
            "week",
            "game_id",
            "game_type",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        },
        "Schedules",
    )
    regular = schedules.filter(pl.col("game_type") == "REG")
    home = regular.select(
        "season",
        "week",
        "game_id",
        pl.col("home_team").alias("team"),
        pl.col("away_team").alias("opponent_team"),
        pl.col("home_score").alias("points_scored"),
    )
    away = regular.select(
        "season",
        "week",
        "game_id",
        pl.col("away_team").alias("team"),
        pl.col("home_team").alias("opponent_team"),
        pl.col("away_score").alias("points_scored"),
    )
    team_games = pl.concat([home, away], how="vertical_relaxed")
    if _has_duplicate_keys(team_games, ["season", "week", "team"]):
        raise ValueError("Schedules contain duplicate team-week games.")
    matchups = team_games.select(
        "season", "week", "team", "opponent_team"
    )
    return matchups, team_games


def _team_history_features(
    team_stats: pl.DataFrame,
    team_games: pl.DataFrame,
    player_weeks: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str], pl.DataFrame, list[str]]:
    _require_columns(
        team_stats,
        {
            "season",
            "week",
            "season_type",
            "game_id",
            "team",
            "opponent_team",
            "passing_yards",
            "rushing_yards",
        },
        "Weekly team stats",
    )
    regular_stats = team_stats.filter(pl.col("season_type") == "REG")
    if _has_duplicate_keys(regular_stats, ["season", "week", "team"]):
        raise ValueError("Weekly team stats contain duplicate team-week rows.")
    if _has_duplicate_keys(team_games, ["season", "week", "team"]):
        raise ValueError("Schedules contain duplicate team-week games.")

    offense_games = regular_stats.join(
        team_games.select("season", "week", "game_id", "team", "points_scored"),
        on=["season", "week", "game_id", "team"],
        how="left",
    )
    if offense_games["points_scored"].null_count():
        raise ValueError("Weekly team stats are missing scheduled scores.")

    opponent_offense = offense_games.select(
        "game_id",
        pl.col("team").alias("opponent_team"),
        pl.col("passing_yards").alias("passing_yards_allowed"),
        pl.col("rushing_yards").alias("rushing_yards_allowed"),
    )
    defense_games = regular_stats.join(
        opponent_offense,
        on=["game_id", "opponent_team"],
        how="left",
    )
    if defense_games.select(
        pl.col("passing_yards_allowed").is_null().any()
        | pl.col("rushing_yards_allowed").is_null().any()
    ).item():
        raise ValueError("Weekly team stats are missing opponent yardage.")

    # Forecast placeholders provide target-week rows, but are still lagged.
    dst_points = player_weeks.filter(
        (pl.col("position") == "DST")
        & pl.col("score_source").is_in(
            ["nflverse_team_stats", "forecast_context"]
        )
    ).select("season", "week", "team", "points_allowed")
    if _has_duplicate_keys(dst_points, ["season", "week", "team"]):
        raise ValueError("D/ST data contains duplicate team-week rows.")
    defense_games = defense_games.join(
        dst_points,
        on=["season", "week", "team"],
        how="left",
    )
    if defense_games["points_allowed"].null_count():
        raise ValueError("D/ST data is missing points allowed for a team game.")

    offense_metrics = {
        "passing_yards": pl.col("passing_yards").cast(pl.Float64).fill_null(0),
        "rushing_yards": pl.col("rushing_yards").cast(pl.Float64).fill_null(0),
        "points_scored": pl.col("points_scored").cast(pl.Float64),
    }
    offense_history, offense_columns = _rolling_features(
        offense_games.with_columns(list(offense_metrics.values())),
        "team",
        OFFENSE_HISTORY_METRICS,
        "opponent_offense",
    )
    defense_metrics = {
        "passing_yards_allowed": pl.col("passing_yards_allowed")
        .cast(pl.Float64)
        .fill_null(0),
        "rushing_yards_allowed": pl.col("rushing_yards_allowed")
        .cast(pl.Float64)
        .fill_null(0),
        "points_allowed": pl.col("points_allowed").cast(pl.Float64),
    }
    defense_history, defense_columns = _rolling_features(
        defense_games.with_columns(list(defense_metrics.values())),
        "team",
        DEFENSE_HISTORY_METRICS,
        "opponent_defense",
    )
    offense_features = offense_history.select(
        "season",
        "week",
        pl.col("team").alias("opponent_team"),
        *offense_columns,
    )
    defense_features = defense_history.select(
        "season",
        "week",
        pl.col("team").alias("opponent_team"),
        *defense_columns,
    )
    return defense_features, defense_columns, offense_features, offense_columns


def build_feature_dataset(
    player_weeks: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
    rankings: pl.DataFrame | None = None,
    player_ids: pl.DataFrame | None = None,
    injuries: pl.DataFrame | None = None,
    output_seasons: Sequence[int] | None = None,
) -> pl.DataFrame:
    """Build pre-week player, matchup, ranking, and injury features."""
    _require_columns(
        player_weeks,
        {
            "season",
            "season_type",
            "week",
            "player_id",
            "player_name",
            "player_display_name",
            "position",
            "team",
            "opponent_team",
            "full_ppr_points",
            "stats_observed",
            "is_bye_week",
            "is_cancelled_game_week",
            "score_source",
            *PLAYER_HISTORY_METRICS.keys(),
            "points_allowed",
        },
        "Player-week data",
    )
    if _has_duplicate_keys(player_weeks, ["season", "week", "player_id"]):
        raise ValueError("Player-week data contains duplicate keys.")
    if player_weeks.filter(
        pl.col("season_type").is_null() | (pl.col("season_type") != "REG")
    ).height:
        raise ValueError("Feature engineering only supports regular-season rows.")

    target_seasons = (
        set(output_seasons)
        if output_seasons is not None
        else set(player_weeks["season"].unique().to_list())
    )
    target_player_weeks = player_weeks.filter(
        pl.col("season").is_in(sorted(target_seasons))
    )
    matchups, team_games = _schedule_context(schedules)
    defense_features, defense_columns, offense_features, offense_columns = (
        _team_history_features(team_stats, team_games, target_player_weeks)
    )
    player_history, player_columns = _player_history_features(player_weeks)
    player_history = player_history.filter(
        pl.col("season").is_in(sorted(target_seasons))
    )
    player_history = player_history.drop("opponent_team").join(
        matchups,
        on=["season", "week", "team"],
        how="left",
    )
    player_history = player_history.join(
        defense_features,
        on=["season", "week", "opponent_team"],
        how="left",
    ).join(
        offense_features,
        on=["season", "week", "opponent_team"],
        how="left",
    )

    context_keys = target_player_weeks.select(
        "season", "week", "player_id"
    ).unique()
    if rankings is not None or player_ids is not None or injuries is not None:
        if rankings is not None and player_ids is None:
            raise ValueError("FantasyPros rankings require a player-ID crosswalk.")
        if rankings is None and player_ids is not None:
            raise ValueError("A player-ID crosswalk requires FantasyPros rankings.")
        cutoffs = _week_cutoffs(schedules)
        if rankings is not None and player_ids is not None:
            context_keys = context_keys.join(
                _rank_features(target_player_weeks, rankings, player_ids, cutoffs),
                on=["season", "week", "player_id"],
                how="left",
            )
        if injuries is not None:
            context_keys = context_keys.join(
                _injury_features(injuries, cutoffs),
                on=["season", "week", "player_id"],
                how="left",
            )
    for column in RANKING_FEATURE_COLUMNS:
        if column not in context_keys.columns:
            context_keys = context_keys.with_columns(
                pl.lit(None, dtype=pl.Float64).alias(column)
                if column == RANKING_FEATURE_COLUMNS[0]
                else (
                    pl.lit(None, dtype=pl.String).alias(column)
                    if column == RANKING_FEATURE_COLUMNS[1]
                    else pl.lit(False).alias(column)
                )
            )
    for column in INJURY_FEATURE_COLUMNS:
        if column not in context_keys.columns:
            context_keys = context_keys.with_columns(
                pl.lit(None, dtype=pl.String).alias(column)
                if column != INJURY_FEATURE_COLUMNS[2]
                else pl.lit(False).alias(column)
            )
    context_keys = context_keys.with_columns(
        pl.col(RANKING_FEATURE_COLUMNS[2]).fill_null(False),
        pl.col(INJURY_FEATURE_COLUMNS[2]).fill_null(False),
    )
    player_history = player_history.join(
        context_keys,
        on=["season", "week", "player_id"],
        how="left",
    ).with_columns(
        pl.col(RANKING_FEATURE_COLUMNS[2]).fill_null(False),
        pl.col(INJURY_FEATURE_COLUMNS[2]).fill_null(False),
    )

    feature_columns = [
        "week",
        "position",
        "team",
        "opponent_team",
        "feature_is_bye_week",
        *RANKING_FEATURE_COLUMNS,
        *INJURY_FEATURE_COLUMNS,
        *player_columns,
        *defense_columns,
        *offense_columns,
    ]
    output = player_history.with_columns(
        pl.col("is_bye_week").fill_null(False).alias("feature_is_bye_week")
    ).select(
        "season",
        "week",
        "player_id",
        "player_name",
        "player_display_name",
        "position",
        "team",
        "opponent_team",
        "is_bye_week",
        "is_cancelled_game_week",
        pl.col("full_ppr_points").alias("target_full_ppr_points"),
        pl.col("stats_observed").alias("target_stats_observed"),
        *[pl.col(column) for column in feature_columns if column not in {
            "week",
            "position",
            "team",
            "opponent_team",
        }],
    )
    if _has_duplicate_keys(output, ["season", "week", "player_id"]):
        raise ValueError("Feature data contains duplicate player-week keys.")
    return output.sort(["season", "week", "position", "player_id"])


def _feature_columns() -> list[str]:
    columns = [
        "week",
        "position",
        "team",
        "opponent_team",
        "feature_is_bye_week",
        "feature_prior_player_weeks",
        "feature_prior_stat_weeks",
    ]
    for feature_name in PLAYER_HISTORY_METRICS.values():
        columns.extend(
            [
                f"feature_previous_week_{feature_name}",
                *[
                    f"feature_{feature_name}_avg_last_{window}_weeks"
                    for window in WINDOWS
                ],
                f"feature_{feature_name}_avg_season_to_date",
            ]
        )
    for prefix, metrics in (
        ("opponent_defense", DEFENSE_HISTORY_METRICS),
        ("opponent_offense", OFFENSE_HISTORY_METRICS),
    ):
        for feature_name in metrics.values():
            columns.extend(
                f"feature_{prefix}_{feature_name}_avg_last_{window}_games"
                for window in WINDOWS
            )
    columns.extend(RANKING_FEATURE_COLUMNS)
    columns.extend(INJURY_FEATURE_COLUMNS)
    return columns


FEATURE_COLUMNS = tuple(_feature_columns())


def build_features(
    seasons: list[int],
    data_dir: Path,
    history_seasons: list[int] | None = None,
) -> int:
    """Build and save features from the cleaned player-week datasets."""
    if not seasons:
        raise ValueError("Provide at least one NFL season.")
    if len(set(seasons)) != len(seasons):
        raise ValueError("Season list contains duplicates.")
    seasons = sorted(seasons)
    history_seasons = (
        sorted(set(history_seasons))
        if history_seasons is not None
        else [seasons[0] - 1]
    )
    if any(season >= seasons[0] for season in history_seasons):
        raise ValueError("History seasons must precede all target seasons.")
    season_label = "-".join(str(season) for season in seasons)
    player_path = data_dir / f"player_weekly_{season_label}.parquet"
    stats_path = data_dir.parent / "raw" / f"weekly_team_stats_{season_label}.parquet"
    schedule_path = data_dir.parent / "raw" / f"schedules_{season_label}.parquet"
    ranking_path = data_dir.parent / "raw" / "historical_fantasy_rankings.parquet"
    player_ids_path = data_dir.parent / "raw" / "fantasy_player_id_crosswalk.parquet"
    injury_path = (
        data_dir.parent
        / "raw"
        / f"weekly_injuries_{season_label}.parquet"
    )
    history_paths = [
        data_dir / f"player_weekly_{season}.parquet"
        for season in history_seasons
    ]
    for path in (
        player_path,
        stats_path,
        schedule_path,
        ranking_path,
        player_ids_path,
        *history_paths,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required feature input not found: {path}")

    if injury_path.exists():
        injuries = pl.read_parquet(injury_path)
    else:
        import nflreadpy as nfl

        injuries = nfl.load_injuries(seasons=seasons)
        if injuries.is_empty():
            raise RuntimeError("The data source returned no weekly injury reports.")
        injury_path.parent.mkdir(parents=True, exist_ok=True)
        injuries.write_parquet(injury_path)

    player_weeks = pl.concat(
        [
            *[pl.read_parquet(path) for path in history_paths],
            pl.read_parquet(player_path),
        ],
        how="diagonal_relaxed",
    )
    features = build_feature_dataset(
        player_weeks,
        pl.read_parquet(stats_path),
        pl.read_parquet(schedule_path),
        pl.read_parquet(ranking_path),
        pl.read_parquet(player_ids_path),
        injuries,
        output_seasons=seasons,
    )
    output_path = data_dir / f"player_week_features_{season_label}.parquet"
    features.write_parquet(output_path)
    metadata = {
        "seasons": seasons,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": features.height,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": "target_full_ppr_points",
        "history_policy": (
            "Lagged and rolling player-performance features use observed stat "
            "rows only; synthetic roster-zero and bye rows remain as zero "
            "targets but are excluded from performance history. Player "
            "history carries across seasons, while season-to-date averages "
            "reset each season. Matchup features use only earlier games in "
            "the current season."
        ),
        "ranking_policy": (
            "Latest same-season FantasyPros positional PPR ranking strictly "
            "before the first kickoff date. Weekly ranks are preferred when "
            "weekly and redraft snapshots share a date; same-day snapshots are "
            "excluded because publication times are unavailable. The source "
            "feature identifies weekly versus redraft rankings."
        ),
        "injury_policy": (
            "Latest report and practice statuses available strictly before "
            "the first scheduled regular-season kickoff of each week. Missing "
            "reports remain unknown."
        ),
    }
    (data_dir / f"feature_metadata_{season_label}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return features.height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="NFL seasons to process, for example: --seasons 2021 2022 2023",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing cleaned data and receiving features",
    )
    parser.add_argument(
        "--history-seasons",
        type=int,
        nargs="+",
        help=(
            "Earlier cleaned seasons used only for player history "
            "(defaults to the season immediately before the first target season)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    row_count = build_features(
        args.seasons,
        args.data_dir,
        history_seasons=args.history_seasons,
    )
    print(f"player_week_features: {row_count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
