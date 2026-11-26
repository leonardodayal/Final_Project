from pathlib import Path

import polars as pl

from src.trades import forecast


def _stub_prediction(
    season: int,
    data_dir: Path,
    model_path: Path,
    output_dir: Path,
) -> dict[str, str]:
    del data_dir, model_path
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"season": [season], "week": [1]}).write_parquet(
        output_dir / f"weekly_predictions_{season}.parquet"
    )
    return {"forecast_mode": "live"}


def test_generate_current_forecast_prepares_missing_public_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    output_dir = tmp_path / "models"
    model_path = output_dir / "weekly_score_model.joblib"
    model_path.parent.mkdir()
    model_path.touch()
    collected = []

    def collect_data(seasons: list[int], destination: Path) -> None:
        collected.append(seasons)
        destination.mkdir(parents=True, exist_ok=True)
        if seasons == list(forecast.HISTORY_SEASONS):
            (
                destination
                / "weekly_player_stats_2021-2022-2023-2024-2025.parquet"
            ).touch()
        else:
            for name in (
                "weekly_player_stats_2026.parquet",
                "historical_fantasy_rankings.parquet",
                "fantasy_player_id_crosswalk.parquet",
            ):
                (destination / name).touch()

    def clean_data(seasons: list[int], raw: Path, processed: Path) -> None:
        assert seasons == list(forecast.HISTORY_SEASONS)
        assert raw == raw_dir
        processed.mkdir(parents=True, exist_ok=True)
        (processed / "player_weekly_2021-2022-2023-2024-2025.parquet").touch()

    monkeypatch.setattr(forecast, "collect_data", collect_data)
    monkeypatch.setattr(forecast, "clean_data", clean_data)
    monkeypatch.setattr(
        forecast,
        "generate_rest_of_season_predictions",
        _stub_prediction,
    )

    result = forecast.generate_current_forecast(
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir,
    )

    assert collected == [list(forecast.HISTORY_SEASONS), [forecast.FORECAST_SEASON]]
    assert result.to_dict(as_series=False) == {"season": [2026], "week": [1]}


def test_generate_current_forecast_uses_cached_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    output_dir = tmp_path / "models"
    model_path = tmp_path / "weekly_score_model.joblib"
    model_path.touch()
    raw_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)
    (processed_dir / "player_weekly_2021-2022-2023-2024-2025.parquet").touch()
    for name in (
        "weekly_player_stats_2026.parquet",
        "historical_fantasy_rankings.parquet",
        "fantasy_player_id_crosswalk.parquet",
    ):
        (raw_dir / name).touch()

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("Cached inputs should not be downloaded or cleaned.")

    monkeypatch.setattr(forecast, "collect_data", fail_if_called)
    monkeypatch.setattr(forecast, "clean_data", fail_if_called)
    monkeypatch.setattr(
        forecast,
        "generate_rest_of_season_predictions",
        _stub_prediction,
    )

    result = forecast.generate_current_forecast(
        data_dir=data_dir,
        model_path=model_path,
        output_dir=output_dir,
    )

    assert result["season"].to_list() == [2026]
