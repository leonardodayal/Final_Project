"""Aggregate weekly player forecasts into rest-of-season values."""

import argparse
from pathlib import Path

import polars as pl

PROJECTION_COLUMN = "predicted_full_ppr_points_if_active"
PLAYER_COLUMNS = {
    "season",
    "week",
    "player_id",
    "player_name",
    "position",
    PROJECTION_COLUMN,
}
POSITION_ALIASES = {
    "D/ST": "DST",
    "DEF": "DST",
    "FB": "RB",
}
VALID_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DST"}


def normalize_position(position: str) -> str:
    normalized = position.strip().upper()
    normalized = POSITION_ALIASES.get(normalized, normalized)
    if normalized not in VALID_POSITIONS:
        raise ValueError(f"Unsupported fantasy position: {position!r}.")
    return normalized


def aggregate_rest_of_season_values(predictions: pl.DataFrame) -> pl.DataFrame:
    """Sum unweighted weekly scores, preserving the selected forecast horizon."""
    missing = PLAYER_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(
            "Weekly predictions are missing columns: "
            + ", ".join(sorted(missing))
        )
    if predictions.is_empty():
        raise ValueError("Weekly predictions contain no player-week rows.")
    if predictions.select(
        pl.any_horizontal(
            pl.col("season").is_null(),
            pl.col("week").is_null(),
            pl.col("player_id").is_null(),
            pl.col("player_name").is_null(),
            pl.col("position").is_null(),
            pl.col(PROJECTION_COLUMN).is_null(),
        ).any()
    ).item():
        raise ValueError("Weekly predictions contain null required values.")
    if predictions["season"].n_unique() != 1:
        raise ValueError("Weekly predictions must contain exactly one season.")
    if predictions.select(
        pl.struct("season", "week", "player_id").is_duplicated().any()
    ).item():
        raise ValueError("Weekly predictions contain duplicate player-week rows.")
    if not predictions[PROJECTION_COLUMN].is_finite().all():
        raise ValueError("Weekly predictions contain non-finite scores.")

    predictions = predictions.with_columns(
        pl.col("position")
        .map_elements(normalize_position, return_dtype=pl.String)
        .alias("position")
    )
    identities = predictions.group_by("player_id").agg(
        pl.col("player_name").n_unique().alias("_names"),
        pl.col("position").n_unique().alias("_positions"),
    )
    if identities.filter(
        (pl.col("_names") != 1) | (pl.col("_positions") != 1)
    ).height:
        raise ValueError("A player has inconsistent name or position values.")

    return (
        predictions.group_by("player_id")
        .agg(
            pl.col("season").first(),
            pl.col("player_name").first(),
            pl.col("position").first(),
            pl.col(PROJECTION_COLUMN).sum().alias("ros_value"),
            pl.col("week").n_unique().alias("forecast_weeks"),
        )
        .sort(["position", "ros_value", "player_name"], descending=[False, True, False])
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    values = aggregate_rest_of_season_values(pl.read_parquet(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    values.write_csv(args.output)
    print(f"player_values: {values.height}")
    print(f"values: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
