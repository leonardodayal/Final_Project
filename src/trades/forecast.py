"""Prepare cached public data and generate 2026 weekly player projections."""

from pathlib import Path

import polars as pl

from src.data.clean import clean_data
from src.data.collect import collect_data
from src.models.predict import generate_rest_of_season_predictions

FORECAST_SEASON = 2026
HISTORY_SEASONS = (2021, 2022, 2023, 2024, 2025)


def generate_current_forecast(
    data_dir: Path = Path("data"),
    model_path: Path = Path("models/weekly_score_model.joblib"),
    output_dir: Path = Path("models"),
    refresh_public_data: bool = False,
) -> pl.DataFrame:
    """Generate live 2026 projections, downloading missing inputs as needed."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    history_label = "-".join(str(season) for season in HISTORY_SEASONS)
    history_stats = raw_dir / f"weekly_player_stats_{history_label}.parquet"
    history_clean = processed_dir / f"player_weekly_{history_label}.parquet"

    if not history_clean.exists():
        if not history_stats.exists():
            collect_data(list(HISTORY_SEASONS), raw_dir)
        clean_data(list(HISTORY_SEASONS), raw_dir, processed_dir)

    current_inputs = (
        raw_dir / f"weekly_player_stats_{FORECAST_SEASON}.parquet",
        raw_dir / "historical_fantasy_rankings.parquet",
        raw_dir / "fantasy_player_id_crosswalk.parquet",
    )
    if refresh_public_data or any(not path.exists() for path in current_inputs):
        collect_data([FORECAST_SEASON], raw_dir)

    metadata = generate_rest_of_season_predictions(
        FORECAST_SEASON,
        data_dir,
        model_path,
        output_dir,
    )
    suffix = "_preseason" if metadata["forecast_mode"] == "strict_preseason_snapshot" else ""
    prediction_path = output_dir / (
        f"weekly_predictions_{FORECAST_SEASON}{suffix}.parquet"
    )
    return pl.read_parquet(prediction_path)
