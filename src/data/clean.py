"""Build a cleaned, ESPN-standard PPR player-week dataset."""

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

PLAYER_POSITIONS = {"QB", "RB", "FB", "WR", "TE", "K"}
ROSTER_STATUSES = {"ACT", "INA", "RES"}
PLAYER_SCORE_COLUMNS = {
    "passing_yards",
    "passing_tds",
    "passing_interceptions",
    "passing_2pt_conversions",
    "rushing_yards",
    "rushing_tds",
    "rushing_2pt_conversions",
    "receptions",
    "receiving_yards",
    "receiving_tds",
    "receiving_2pt_conversions",
    "fumbles_lost_total",
    "special_teams_tds",
    "fumble_recovery_tds",
    "fg_made_0_19",
    "fg_made_20_29",
    "fg_made_30_39",
    "fg_made_40_49",
    "fg_made_50_59",
    "fg_made_60_",
    "fg_missed",
    "pat_made",
}
DST_SCORE_COLUMNS = {
    "def_sacks",
    "def_interceptions",
    "fumble_recovery_opp",
    "def_tds",
    "special_teams_tds",
    "def_safeties",
    "def_punt_blocks",
    "def_pat_blocks",
    "def_fg_blocks",
}
CANCELLED_GAME_TEAM_WEEKS = [
    {"season": 2022, "week": 17, "team": "BUF"},
    {"season": 2022, "week": 17, "team": "CIN"},
]


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


def calculate_player_scores(stats: pl.DataFrame) -> pl.DataFrame:
    """Calculate standard full-PPR scores from weekly stat components."""
    _require_columns(stats, PLAYER_SCORE_COLUMNS, "Player stats")
    values = {
        name: pl.col(name).cast(pl.Float64).fill_null(0)
        for name in PLAYER_SCORE_COLUMNS
    }
    offensive_score = (
        values["passing_yards"] / 25
        + 4 * values["passing_tds"]
        - 2 * values["passing_interceptions"]
        + 2 * values["passing_2pt_conversions"]
        + values["rushing_yards"] / 10
        + 6 * values["rushing_tds"]
        + 2 * values["rushing_2pt_conversions"]
        + values["receptions"]
        + values["receiving_yards"] / 10
        + 6 * values["receiving_tds"]
        + 2 * values["receiving_2pt_conversions"]
        - 2 * values["fumbles_lost_total"]
        + 6 * values["special_teams_tds"]
        + 6 * values["fumble_recovery_tds"]
    )
    kicker_score = (
        3
        * (
            values["fg_made_0_19"]
            + values["fg_made_20_29"]
            + values["fg_made_30_39"]
        )
        + 4 * values["fg_made_40_49"]
        + 5 * values["fg_made_50_59"]
        + 6 * values["fg_made_60_"]
        - values["fg_missed"]
        + values["pat_made"]
    )
    return stats.with_columns(
        (offensive_score + kicker_score).round(2).alias("full_ppr_points")
    )


def _points_allowed_score(points: pl.Expr) -> pl.Expr:
    return (
        pl.when(points == 0)
        .then(5)
        .when(points <= 6)
        .then(4)
        .when(points <= 13)
        .then(3)
        .when(points <= 17)
        .then(1)
        .when(points <= 27)
        .then(0)
        .when(points <= 34)
        .then(-1)
        .when(points <= 45)
        .then(-3)
        .otherwise(-5)
    )


def _cancelled_game_weeks() -> pl.DataFrame:
    return pl.DataFrame(
        CANCELLED_GAME_TEAM_WEEKS,
        schema={"season": pl.Int64, "week": pl.Int64, "team": pl.String},
    ).with_columns(pl.lit(True).alias("is_cancelled_game_week"))


def _clean_players(
    stats: pl.DataFrame,
    rosters: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        stats,
        PLAYER_SCORE_COLUMNS
        | {
            "season",
            "week",
            "season_type",
            "player_id",
            "player_name",
            "player_display_name",
            "position",
            "team",
            "fantasy_points_ppr",
        },
        "Player stats",
    )
    _require_columns(
        rosters,
        {
            "season",
            "week",
            "game_type",
            "gsis_id",
            "position",
            "status",
            "team",
            "full_name",
        },
        "Weekly rosters",
    )
    _require_columns(
        schedules,
        {"season", "week", "game_type", "home_team", "away_team"},
        "Schedules",
    )

    regular_stats = stats.filter(
        (pl.col("season_type") == "REG")
        & pl.col("position").is_in(PLAYER_POSITIONS)
    ).rename({"position": "source_position"})
    regular_rosters = (
        rosters.filter(
            (pl.col("game_type") == "REG")
            & pl.col("status").is_in(ROSTER_STATUSES)
            & pl.col("position").is_in(PLAYER_POSITIONS)
        )
        .select(
            "season",
            "week",
            pl.col("gsis_id").alias("roster_player_id"),
            pl.col("position").alias("roster_position"),
            pl.col("team").alias("roster_team"),
            pl.col("full_name").alias("roster_name"),
            pl.col("status").alias("roster_status"),
        )
    )
    if regular_rosters.select(
        pl.struct("season", "week", "roster_player_id").is_duplicated().any()
    ).item():
        raise ValueError("Weekly rosters contain duplicate player-week keys.")

    roster_metadata = regular_rosters.select(
        "season",
        "week",
        "roster_player_id",
        "roster_position",
        "roster_team",
        "roster_name",
        "roster_status",
    )
    joined = regular_stats.join(
        roster_metadata,
        left_on=["season", "week", "player_id"],
        right_on=["season", "week", "roster_player_id"],
        how="left",
    )
    joined = joined.with_columns(
        pl.coalesce("team", "roster_team").alias("team"),
        pl.coalesce("source_position", "roster_position").alias("source_position"),
        pl.coalesce("player_display_name", "roster_name").alias(
            "player_display_name"
        ),
        pl.lit(True).alias("stats_observed"),
        pl.lit("nflverse_player_stats").alias("score_source"),
    )

    roster_only = regular_rosters.join(
        regular_stats.select("season", "week", "player_id"),
        left_on=["season", "week", "roster_player_id"],
        right_on=["season", "week", "player_id"],
        how="anti",
    )
    zero_expressions = []
    for column, dtype in regular_stats.schema.items():
        if column in {"season", "week"}:
            zero_expressions.append(pl.col(column))
        elif column == "season_type":
            zero_expressions.append(pl.lit("REG").alias(column))
        elif column == "player_id":
            zero_expressions.append(
                pl.col("roster_player_id").cast(dtype).alias(column)
            )
        elif column == "player_name":
            zero_expressions.append(pl.col("roster_name").alias(column))
        elif column == "player_display_name":
            zero_expressions.append(pl.col("roster_name").alias(column))
        elif column == "source_position":
            zero_expressions.append(pl.col("roster_position").alias(column))
        elif column == "team":
            zero_expressions.append(pl.col("roster_team").alias(column))
        elif column in PLAYER_SCORE_COLUMNS:
            zero_expressions.append(pl.lit(0).cast(dtype).alias(column))
        else:
            zero_expressions.append(pl.lit(None, dtype=dtype).alias(column))

    zero_rows = roster_only.select(
        *zero_expressions,
        pl.col("roster_status"),
        pl.lit(False).alias("stats_observed"),
        pl.lit("roster_zero").alias("score_source"),
    )
    players = pl.concat(
        [
            joined.select(
                *regular_stats.columns,
                "roster_status",
                "stats_observed",
                "score_source",
            ),
            zero_rows,
        ],
        how="vertical_relaxed",
    ).with_columns(
        pl.when(pl.col("source_position") == "FB")
        .then(pl.lit("RB"))
        .otherwise(pl.col("source_position"))
        .alias("position"),
        *[
            pl.col(column).cast(pl.Float64).fill_null(0).alias(column)
            for column in PLAYER_SCORE_COLUMNS
        ],
    )
    players = calculate_player_scores(players).with_columns(
        (
            pl.col("full_ppr_points")
            - pl.col("fantasy_points_ppr").cast(pl.Float64)
        )
        .round(2)
        .alias("source_ppr_difference")
    )

    schedule_teams = (
        schedules.filter(pl.col("game_type") == "REG")
        .select(
            "season",
            "week",
            pl.col("home_team").alias("team"),
        )
        .vstack(
            schedules.filter(pl.col("game_type") == "REG").select(
                "season",
                "week",
                pl.col("away_team").alias("team"),
            )
        )
        .unique()
        .with_columns(pl.lit(True).alias("team_played"))
    )
    cancelled_weeks = _cancelled_game_weeks()
    return (
        players.join(schedule_teams, on=["season", "week", "team"], how="left")
        .join(cancelled_weeks, on=["season", "week", "team"], how="left")
        .with_columns(
            pl.col("is_cancelled_game_week").fill_null(False),
            (
                pl.col("team").is_not_null()
                & pl.col("team_played").is_null()
                & pl.col("is_cancelled_game_week").is_null()
            ).alias("is_bye_week"),
        )
        .drop("team_played")
    )


def _clean_defenses(
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    _require_columns(
        team_stats,
        DST_SCORE_COLUMNS
        | {"season", "week", "season_type", "game_id", "team", "opponent_team"},
        "Weekly team stats",
    )
    _require_columns(
        schedules,
        {
            "season",
            "week",
            "game_type",
            "game_id",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        },
        "Schedules",
    )

    regular_stats = team_stats.filter(pl.col("season_type") == "REG")
    regular_schedules = schedules.filter(pl.col("game_type") == "REG")
    scheduled_teams = regular_schedules.select(
        "season",
        "week",
        "game_id",
        pl.col("home_team").alias("team"),
    ).vstack(
        regular_schedules.select(
            "season",
            "week",
            "game_id",
            pl.col("away_team").alias("team"),
        )
    )
    actual_teams = regular_stats.select("season", "week", "game_id", "team")
    missing_team_stats = scheduled_teams.join(
        actual_teams,
        on=["season", "week", "game_id", "team"],
        how="anti",
    )
    if not missing_team_stats.is_empty():
        raise ValueError("Weekly team stats are missing scheduled team-game rows.")

    opponent_defense = regular_stats.select(
        "game_id",
        pl.col("team").alias("opponent_team"),
        pl.col("def_tds").alias("opponent_def_tds"),
    )
    defenses = regular_stats.join(
        opponent_defense,
        on=["game_id", "opponent_team"],
        how="left",
    ).join(
        regular_schedules.select(
            "game_id", "home_team", "away_team", "home_score", "away_score"
        ),
        on="game_id",
        how="left",
    )
    points_allowed = (
        pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score"))
        - 6 * pl.col("opponent_def_tds").fill_null(0)
    ).cast(pl.Int64)
    defenses = defenses.with_columns(
        points_allowed.alias("points_allowed")
    ).with_columns(
        (
            pl.col("def_sacks").fill_null(0)
            + 2 * pl.col("def_interceptions").fill_null(0)
            + 2 * pl.col("fumble_recovery_opp").fill_null(0)
            + 6 * pl.col("def_tds").fill_null(0)
            + 6 * pl.col("special_teams_tds").fill_null(0)
            + 2
            * (
                pl.col("def_punt_blocks").fill_null(0)
                + pl.col("def_pat_blocks").fill_null(0)
                + pl.col("def_fg_blocks").fill_null(0)
            )
            + 2 * pl.col("def_safeties").fill_null(0)
            + _points_allowed_score(pl.col("points_allowed"))
        )
        .round(2)
        .alias("full_ppr_points")
    )

    defense_games = defenses.select(
        "season",
        "week",
        pl.lit("REG").alias("season_type"),
        pl.concat_str([pl.lit("DST:"), pl.col("team")]).alias("player_id"),
        pl.concat_str([pl.col("team"), pl.lit(" D/ST")]).alias("player_name"),
        pl.concat_str([pl.col("team"), pl.lit(" D/ST")]).alias(
            "player_display_name"
        ),
        pl.lit("DST").alias("source_position"),
        pl.lit("DST").alias("position"),
        "team",
        pl.lit(None, dtype=pl.Float64).alias("fantasy_points_ppr"),
        "full_ppr_points",
        pl.lit(None, dtype=pl.Float64).alias("source_ppr_difference"),
        pl.lit(None, dtype=pl.String).alias("roster_status"),
        pl.lit(True).alias("stats_observed"),
        pl.lit("nflverse_team_stats").alias("score_source"),
        pl.lit(False).alias("is_bye_week"),
        pl.lit(False).alias("is_cancelled_game_week"),
        "points_allowed",
    )

    season_weeks = regular_schedules.select("season", "week").unique()
    season_teams = scheduled_teams.select("season", "team").unique()
    full_team_weeks = season_weeks.join(season_teams, on="season", how="inner")
    bye_weeks = full_team_weeks.join(
        defense_games.select("season", "week", "team"),
        on=["season", "week", "team"],
        how="anti",
    ).join(_cancelled_game_weeks(), on=["season", "week", "team"], how="left")
    bye_rows = bye_weeks.select(
        "season",
        "week",
        pl.lit("REG").alias("season_type"),
        pl.concat_str([pl.lit("DST:"), pl.col("team")]).alias("player_id"),
        pl.concat_str([pl.col("team"), pl.lit(" D/ST")]).alias("player_name"),
        pl.concat_str([pl.col("team"), pl.lit(" D/ST")]).alias(
            "player_display_name"
        ),
        pl.lit("DST").alias("source_position"),
        pl.lit("DST").alias("position"),
        "team",
        pl.lit(None, dtype=pl.Float64).alias("fantasy_points_ppr"),
        pl.lit(0.0).alias("full_ppr_points"),
        pl.lit(None, dtype=pl.Float64).alias("source_ppr_difference"),
        pl.lit(None, dtype=pl.String).alias("roster_status"),
        pl.lit(False).alias("stats_observed"),
        pl.when(pl.col("is_cancelled_game_week").fill_null(False))
        .then(pl.lit("cancelled_game_zero"))
        .otherwise(pl.lit("bye_zero"))
        .alias("score_source"),
        pl.col("is_cancelled_game_week").is_null().alias("is_bye_week"),
        pl.col("is_cancelled_game_week").fill_null(False).alias(
            "is_cancelled_game_week"
        ),
        pl.lit(None, dtype=pl.Int64).alias("points_allowed"),
    )
    return pl.concat([defense_games, bye_rows], how="vertical_relaxed")


def build_player_week_dataset(
    stats: pl.DataFrame,
    rosters: pl.DataFrame,
    team_stats: pl.DataFrame,
    schedules: pl.DataFrame,
) -> pl.DataFrame:
    """Create regular-season player and D/ST weekly scoring rows."""
    players = _clean_players(stats, rosters, schedules)
    defenses = _clean_defenses(team_stats, schedules)
    dataset = pl.concat([players, defenses], how="diagonal_relaxed")
    if dataset.select(
        pl.struct("season", "week", "player_id").is_duplicated().any()
    ).item():
        raise ValueError("Cleaned data contains duplicate player-week keys.")
    return dataset.sort(["season", "week", "position", "player_id"])


def clean_data(seasons: list[int], input_dir: Path, output_dir: Path) -> int:
    """Load nflverse context data and save the cleaned player-week Parquet."""
    if not seasons:
        raise ValueError("Provide at least one NFL season.")
    season_label = "-".join(str(season) for season in seasons)
    stats_path = input_dir / f"weekly_player_stats_{season_label}.parquet"
    if not stats_path.exists():
        raise FileNotFoundError(f"Weekly player stats not found: {stats_path}")

    import nflreadpy as nfl

    rosters = nfl.load_rosters_weekly(seasons=seasons)
    team_stats = nfl.load_team_stats(seasons=seasons, summary_level="week")
    schedules = nfl.load_schedules(seasons=seasons)
    for name, frame in (
        ("weekly_rosters", rosters),
        ("weekly_team_stats", team_stats),
        ("schedules", schedules),
    ):
        frame.write_parquet(input_dir / f"{name}_{season_label}.parquet")

    dataset = build_player_week_dataset(
        pl.read_parquet(stats_path),
        rosters,
        team_stats,
        schedules,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"player_weekly_{season_label}.parquet"
    dataset.write_parquet(output_path)
    metadata = {
        "source": "nflverse via nflreadpy",
        "nflreadpy_version": nfl.__version__,
        "seasons": seasons,
        "scoring": "ESPN standard full PPR",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": dataset.height,
        "player_rows": dataset.filter(pl.col("position") != "DST").height,
        "defense_rows": dataset.filter(pl.col("position") == "DST").height,
        "roster_zero_rows": dataset.filter(
            pl.col("score_source") == "roster_zero"
        ).height,
        "bye_zero_rows": dataset.filter(
            pl.col("score_source") == "bye_zero"
        ).height,
        "cancelled_game_zero_rows": dataset.filter(
            pl.col("score_source") == "cancelled_game_zero"
        ).height,
        "rankings_included": False,
    }
    (output_dir / f"cleaning_metadata_{season_label}.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset.height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        required=True,
        help="NFL seasons to clean, for example: --seasons 2021 2022 2023",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing collected player stats (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for cleaned data (default: data/processed)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    row_count = clean_data(args.seasons, args.input_dir, args.output_dir)
    print(f"player_week_dataset: {row_count:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
