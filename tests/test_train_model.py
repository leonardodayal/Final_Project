import pytest
import polars as pl
from sklearn.exceptions import ConvergenceWarning

from src.features.build import FEATURE_COLUMNS
from src.models.train import TARGET_COLUMN, train_and_evaluate


CATEGORICAL_FEATURES = {
    "position",
    "team",
    "opponent_team",
    "feature_fantasypros_rank_source",
    "feature_injury_report_status",
    "feature_injury_practice_status",
}
BOOLEAN_FEATURES = {
    "feature_is_bye_week",
    "feature_fantasypros_rank_available",
    "feature_injury_report_available",
}


def _feature_row(season: int, index: int) -> dict:
    position = "RB" if index % 2 == 0 else "WR"
    row = {
        "season": season,
        "week": index + 1,
        "player_id": f"{season}-{index}",
        "player_name": f"Player {index}",
        "player_display_name": f"Player {index}",
        "position": position,
        "team": "BUF" if index % 2 == 0 else "MIA",
        "opponent_team": "MIA" if index % 2 == 0 else "BUF",
        "is_bye_week": False,
        "is_cancelled_game_week": False,
        TARGET_COLUMN: float(index + season % 10),
    }
    for column in FEATURE_COLUMNS:
        if column in CATEGORICAL_FEATURES:
            row[column] = {
                "position": position,
                "team": row["team"],
                "opponent_team": row["opponent_team"],
                "feature_fantasypros_rank_source": (
                    "weekly" if index % 2 == 0 else "redraft"
                ),
                "feature_injury_report_status": (
                    "Questionable" if index % 3 == 0 else None
                ),
                "feature_injury_practice_status": (
                    "Full Participation in Practice"
                    if index % 3 == 0
                    else None
                ),
            }[column]
        elif column in BOOLEAN_FEATURES:
            row[column] = index % 2 == 0
        elif column == "week":
            continue
        else:
            row[column] = float(index + 1)
    return row


def test_training_uses_chronological_splits_and_saves_results(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr("src.models.train.MLP_MAX_ITERATIONS", 1)
    feature_rows = [
        _feature_row(season, index)
        for season in range(2021, 2026)
        for index in range(4)
    ]
    feature_path = tmp_path / "features.parquet"
    pl.DataFrame(feature_rows).write_parquet(feature_path)

    with pytest.warns(ConvergenceWarning):
        report = train_and_evaluate(feature_path, tmp_path / "models")

    assert report["seasons"] == {
        "training": [2021, 2022, 2023],
        "validation": 2024,
        "test": 2025,
    }
    assert report["test"]["selected_model_metrics"]["overall"]["rows"] == 4
    assert (tmp_path / "models" / "weekly_score_model.joblib").exists()
    assert (tmp_path / "models" / "step4_metrics.json").exists()
    predictions = pl.read_parquet(tmp_path / "models" / "step4_predictions.parquet")
    assert predictions.group_by("split").len().sort("split")["len"].to_list() == [
        4,
        4,
    ]
