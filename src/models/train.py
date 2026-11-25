"""Train and evaluate leakage-safe weekly fantasy-score regression models."""

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build import FEATURE_COLUMNS, INJURY_FEATURE_COLUMNS
from src.models.roles import (
    BACKUP_SCORE_MULTIPLIER,
    STARTER_COUNTS,
    apply_starter_rule,
)

TARGET_COLUMN = "target_full_ppr_points"
OBSERVED_COLUMN = "target_stats_observed"
TRAIN_SEASONS = (2021, 2022, 2023)
VALIDATION_SEASON = 2024
TEST_SEASON = 2025
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
MLP_HIDDEN_LAYERS = (32,)
MLP_ALPHA = 0.001
MLP_MAX_ITERATIONS = 100
RANDOM_STATE = 42
CATEGORICAL_FEATURES = (
    "position",
    "team",
    "opponent_team",
    "feature_fantasypros_rank_source",
    "feature_injury_report_status",
    "feature_injury_practice_status",
)


def _validate_feature_data(frame: pl.DataFrame) -> None:
    required = {
        "season",
        "week",
        "player_id",
        "position",
        TARGET_COLUMN,
        OBSERVED_COLUMN,
        *FEATURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Feature data is missing required columns: "
            + ", ".join(sorted(missing))
        )
    if frame.select(
        pl.struct(["season", "week", "player_id"]).is_duplicated().any()
    ).item():
        raise ValueError("Feature data contains duplicate player-week keys.")
    if frame[TARGET_COLUMN].null_count():
        raise ValueError("Training targets cannot be null.")
    if frame[OBSERVED_COLUMN].null_count():
        raise ValueError("Observed-stat labels cannot be null.")
    if frame.schema[OBSERVED_COLUMN] != pl.Boolean:
        raise ValueError("Observed-stat labels must be Boolean.")
    target_values = frame[TARGET_COLUMN].cast(pl.Float64).to_numpy()
    if not np.isfinite(target_values).all():
        raise ValueError("Training targets must be finite numbers.")


def _split_by_season(
    frame: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    splits = []
    for label, seasons in (
        ("training", TRAIN_SEASONS),
        ("validation", (VALIDATION_SEASON,)),
        ("test", (TEST_SEASON,)),
    ):
        subset = frame.filter(pl.col("season").is_in(seasons))
        if subset.is_empty():
            raise ValueError(
                f"Feature data has no rows for the {label} season(s): "
                + ", ".join(str(season) for season in seasons)
            )
        splits.append(subset)
    return splits[0], splits[1], splits[2]


def _feature_frame(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: frame.get_column(column).to_list()
            for column in feature_columns
        }
    )


def _preprocessor(feature_columns: Sequence[str]) -> ColumnTransformer:
    categorical = [
        column for column in CATEGORICAL_FEATURES if column in feature_columns
    ]
    numeric = [
        column for column in feature_columns if column not in categorical
    ]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="Unknown"),
            ),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        sparse_threshold=0.0,
    )


def _model_pipeline(
    feature_columns: Sequence[str],
    family: str,
    alpha: float,
) -> Pipeline:
    if family == "ridge":
        regressor = Ridge(alpha=alpha)
    elif family == "mlp":
        regressor = MLPRegressor(
            hidden_layer_sizes=MLP_HIDDEN_LAYERS,
            activation="relu",
            solver="adam",
            alpha=alpha,
            learning_rate_init=0.001,
            max_iter=MLP_MAX_ITERATIONS,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported regression model family: {family}")
    return Pipeline(
        [
            ("preprocessor", _preprocessor(feature_columns)),
            ("regressor", regressor),
        ]
    )


def _model_specifications() -> list[dict]:
    specifications = []
    for use_injury_features in (True, False):
        for alpha in RIDGE_ALPHAS:
            specifications.append(
                {
                    "family": "ridge",
                    "alpha": alpha,
                    "use_injury_features": use_injury_features,
                    "name": (
                        f"ridge_alpha_{alpha:g}_"
                        f"{'with' if use_injury_features else 'without'}_injury"
                    ),
                }
            )
        specifications.append(
            {
                "family": "mlp",
                "alpha": MLP_ALPHA,
                "use_injury_features": use_injury_features,
                "name": (
                    f"mlp_32_alpha_{MLP_ALPHA:g}_"
                    f"{'with' if use_injury_features else 'without'}_injury"
                ),
            }
        )
    return specifications


def _active_features(use_injury_features: bool) -> tuple[str, ...]:
    if use_injury_features:
        return FEATURE_COLUMNS
    return tuple(
        column
        for column in FEATURE_COLUMNS
        if column not in INJURY_FEATURE_COLUMNS
    )


def _position_means(reference: pl.DataFrame) -> tuple[dict[str, float], float]:
    means = reference.group_by("position").agg(
        pl.col(TARGET_COLUMN).mean().alias("mean_score")
    )
    by_position = {
        row["position"]: float(row["mean_score"])
        for row in means.to_dicts()
    }
    return by_position, float(reference[TARGET_COLUMN].mean())


def _baseline_predictions(
    reference: pl.DataFrame,
    target: pl.DataFrame,
) -> dict[str, list[float]]:
    by_position, overall_mean = _position_means(reference)
    positions = target["position"].to_list()

    def fill_missing(column: str) -> list[float]:
        values = target[column].cast(pl.Float64, strict=False).to_list()
        return [
            float(value)
            if value is not None and np.isfinite(value)
            else by_position.get(position, overall_mean)
            for value, position in zip(values, positions, strict=True)
        ]

    return {
        "position_mean": [
            by_position.get(position, overall_mean) for position in positions
        ],
        "previous_week": fill_missing("feature_previous_week_score"),
        "rolling_3_week": fill_missing("feature_score_avg_last_3_weeks"),
        "rolling_5_week": fill_missing("feature_score_avg_last_5_weeks"),
    }


def _position_adjustments(
    actual: np.ndarray,
    predicted: np.ndarray,
    positions: Sequence[str],
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    position_values = np.asarray(positions, dtype=object)
    return {
        str(position): float(
            np.mean(
                actual[position_values == position]
                - predicted[position_values == position]
            )
        )
        for position in sorted(set(position_values))
    }


def _adjust_scores(
    predicted: np.ndarray,
    positions: Sequence[str],
    adjustments: dict[str, float],
) -> np.ndarray:
    position_values = np.asarray(positions, dtype=object)
    calibrated = np.asarray(predicted, dtype=float) + np.array(
        [adjustments.get(str(position), 0.0) for position in position_values]
    )
    quarterbacks = position_values == "QB"
    return np.where(quarterbacks, np.maximum(calibrated, 0.0), calibrated)


def _starter_weighted_baselines(
    baselines: dict[str, list[float]],
    frame: pl.DataFrame,
    starter_counts: dict[str, int],
    backup_multiplier: float,
) -> dict[str, list[float]]:
    return {
        name: apply_starter_rule(
            frame,
            np.asarray(values, dtype=float),
            starter_counts,
            backup_multiplier,
        )[0].tolist()
        for name, values in baselines.items()
    }


def _metric_summary(
    actual: np.ndarray,
    predicted: np.ndarray,
    positions: Sequence[str],
) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    position_values = np.asarray(positions, dtype=object)

    def calculate(mask: np.ndarray) -> dict[str, float | int]:
        return {
            "rows": int(mask.sum()),
            "mae": float(mean_absolute_error(actual[mask], predicted[mask])),
            "rmse": float(np.sqrt(mean_squared_error(actual[mask], predicted[mask]))),
        }

    all_rows = np.ones(actual.shape[0], dtype=bool)
    result = {"overall": calculate(all_rows), "by_position": {}}
    for position in sorted(set(position_values)):
        mask = position_values == position
        result["by_position"][str(position)] = calculate(mask)
    return result


def _prediction_frame(
    frame: pl.DataFrame,
    split_name: str,
    actual: np.ndarray,
    selected_prediction: np.ndarray,
    conditional_prediction: np.ndarray,
    predicted_starter: np.ndarray,
    starter_multiplier: np.ndarray,
    baselines: dict[str, list[float]],
) -> pl.DataFrame:
    result = frame.select("season", "week", "player_id", "position").with_columns(
        pl.lit(split_name).alias("split"),
        pl.Series("actual_full_ppr_points", actual),
        pl.Series("selected_model_prediction", selected_prediction),
        pl.Series("conditional_score_prediction", conditional_prediction),
        pl.Series("predicted_starter", predicted_starter),
        pl.Series("starter_score_multiplier", starter_multiplier),
    )
    for name, prediction in baselines.items():
        result = result.with_columns(
            pl.Series(f"baseline_{name}", prediction)
        )
    return result


def train_and_evaluate(
    feature_path: Path,
    output_dir: Path,
) -> dict:
    """Train candidates, select on 2024, and evaluate once on 2025."""
    frame = pl.read_parquet(feature_path)
    _validate_feature_data(frame)
    train_frame, validation_frame, test_frame = _split_by_season(frame)
    train_observed = train_frame.filter(pl.col(OBSERVED_COLUMN))
    validation_observed = validation_frame.filter(pl.col(OBSERVED_COLUMN))
    if train_observed.is_empty() or validation_observed.is_empty():
        raise ValueError(
            "Training and validation data must contain observed-stat rows."
        )
    if train_frame[OBSERVED_COLUMN].n_unique() != 2:
        raise ValueError(
            "Training data must include both observed and unobserved weeks."
        )

    validation_y = validation_observed[TARGET_COLUMN].to_numpy()
    validation_positions = validation_observed["position"].to_list()
    validation_mask = validation_frame[OBSERVED_COLUMN].to_numpy().astype(bool)
    candidate_results = []

    for specification in _model_specifications():
        feature_columns = _active_features(
            specification["use_injury_features"]
        )
        model = _model_pipeline(
            feature_columns,
            specification["family"],
            specification["alpha"],
        )
        model.fit(
            _feature_frame(train_observed, feature_columns),
            train_observed[TARGET_COLUMN].to_numpy(),
        )
        prediction = model.predict(
            _feature_frame(validation_frame, feature_columns)
        )
        observed_prediction = prediction[validation_mask]
        candidate_results.append(
            {
                "specification": specification,
                "feature_columns": feature_columns,
                "model": model,
                "validation_prediction": prediction,
                "metrics": _metric_summary(
                    validation_y,
                    observed_prediction,
                    validation_positions,
                ),
            }
        )

    selected = min(
        candidate_results,
        key=lambda result: result["metrics"]["overall"]["mae"],
    )
    selected_specification = selected["specification"]
    selected_features = selected["feature_columns"]
    selected_validation_prediction = selected["validation_prediction"]
    score_adjustments = _position_adjustments(
        validation_y,
        selected_validation_prediction[validation_mask],
        validation_positions,
    )
    validation_conditional_prediction = _adjust_scores(
        selected_validation_prediction,
        validation_frame["position"].to_list(),
        score_adjustments,
    )
    (
        validation_prediction,
        validation_conditional_prediction,
        validation_starters,
        validation_multipliers,
    ) = apply_starter_rule(
        validation_frame,
        validation_conditional_prediction,
        STARTER_COUNTS,
        BACKUP_SCORE_MULTIPLIER,
    )
    validation_baselines = _starter_weighted_baselines(
        _baseline_predictions(train_observed, validation_frame),
        validation_frame,
        STARTER_COUNTS,
        BACKUP_SCORE_MULTIPLIER,
    )
    validation_metrics = {
        name: _metric_summary(
            validation_frame[TARGET_COLUMN].to_numpy(),
            np.asarray(prediction),
            validation_frame["position"].to_list(),
        )
        for name, prediction in validation_baselines.items()
    }
    validation_expected_metrics = _metric_summary(
        validation_frame[TARGET_COLUMN].to_numpy(),
        validation_prediction,
        validation_frame["position"].to_list(),
    )
    validation_conditional_metrics = _metric_summary(
        validation_y,
        validation_conditional_prediction[validation_mask],
        validation_positions,
    )

    train_validation_frame = pl.concat(
        [train_frame, validation_frame],
        how="vertical_relaxed",
    )
    train_validation_observed = train_validation_frame.filter(
        pl.col(OBSERVED_COLUMN)
    )
    final_model = _model_pipeline(
        selected_features,
        selected_specification["family"],
        selected_specification["alpha"],
    )
    final_model.fit(
        _feature_frame(train_validation_observed, selected_features),
        train_validation_observed[TARGET_COLUMN].to_numpy(),
    )
    test_raw_prediction = final_model.predict(
        _feature_frame(test_frame, selected_features)
    )
    test_calibrated_prediction = _adjust_scores(
        test_raw_prediction,
        test_frame["position"].to_list(),
        score_adjustments,
    )
    (
        test_prediction,
        test_conditional_prediction,
        test_starters,
        test_multipliers,
    ) = apply_starter_rule(
        test_frame,
        test_calibrated_prediction,
        STARTER_COUNTS,
        BACKUP_SCORE_MULTIPLIER,
    )
    test_y = test_frame[TARGET_COLUMN].to_numpy()
    test_positions = test_frame["position"].to_list()
    test_observed = test_frame.filter(pl.col(OBSERVED_COLUMN))
    test_mask = test_frame[OBSERVED_COLUMN].to_numpy().astype(bool)
    test_baselines = _starter_weighted_baselines(
        _baseline_predictions(train_validation_observed, test_frame),
        test_frame,
        STARTER_COUNTS,
        BACKUP_SCORE_MULTIPLIER,
    )

    best_validation_baseline = min(
        validation_metrics,
        key=lambda name: validation_metrics[name]["overall"]["mae"],
    )
    test_conditional_metrics = _metric_summary(
        test_observed[TARGET_COLUMN].to_numpy(),
        test_conditional_prediction[test_mask],
        test_observed["position"].to_list(),
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_path": str(feature_path),
        "target_column": TARGET_COLUMN,
        "target_policy": (
            "The score model trains on observed-stat weeks only. For each "
            "team-week, top-ranked players by position receive their full "
            "if-active score and other players receive half."
        ),
        "seasons": {
            "training": list(TRAIN_SEASONS),
            "validation": VALIDATION_SEASON,
            "test": TEST_SEASON,
        },
        "feature_count": len(FEATURE_COLUMNS),
        "starter_rule": {
            "rank_feature": "feature_fantasypros_position_rank",
            "starter_counts_per_team": STARTER_COUNTS,
            "backup_score_multiplier": BACKUP_SCORE_MULTIPLIER,
            "unranked_fallback": (
                "Unranked players fill remaining starter slots by descending "
                "conditional score."
            ),
        },
        "conditional_score_adjustments": score_adjustments,
        "selected_model": {
            **selected_specification,
            "feature_columns": list(selected_features),
            "beats_best_validation_baseline": (
                validation_expected_metrics["overall"]["mae"]
                < validation_metrics[best_validation_baseline]["overall"]["mae"]
            ),
        },
        "validation": {
            "best_baseline": best_validation_baseline,
            "baseline_metrics": validation_metrics,
            "candidate_models": [
                {
                    **result["specification"],
                    "conditional_score_metrics": result["metrics"]["overall"],
                }
                for result in candidate_results
            ],
            "conditional_score_metrics": validation_conditional_metrics,
            "selected_model_metrics": validation_expected_metrics,
        },
        "test": {
            "baseline_metrics": {
                name: _metric_summary(
                    test_y,
                    np.asarray(prediction),
                    test_positions,
                )
                for name, prediction in test_baselines.items()
            },
            "conditional_score_metrics": test_conditional_metrics,
            "selected_model_metrics": _metric_summary(
                test_y,
                test_prediction,
                test_positions,
            ),
        },
        "preprocessing": (
            "Numeric fields use training-set median imputation and standard "
            "scaling; categorical fields use training-set one-hot encoding. "
            "Injury features are ablated as a model-selection comparison."
        ),
        "model_library_versions": {
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
        },
        "random_state": RANDOM_STATE,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "weekly_score_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_columns": list(selected_features),
            "target_column": TARGET_COLUMN,
            "selected_model": selected_specification,
            "conditional_score_adjustments": score_adjustments,
            "starter_counts": STARTER_COUNTS,
            "backup_score_multiplier": BACKUP_SCORE_MULTIPLIER,
        },
        artifact_path,
    )
    report["model_artifact"] = str(artifact_path)
    metrics_path = output_dir / "step4_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions = pl.concat(
        [
            _prediction_frame(
                validation_frame,
                "validation",
                validation_frame[TARGET_COLUMN].to_numpy(),
                validation_prediction,
                validation_conditional_prediction,
                validation_starters,
                validation_multipliers,
                validation_baselines,
            ),
            _prediction_frame(
                test_frame,
                "test",
                test_y,
                test_prediction,
                test_conditional_prediction,
                test_starters,
                test_multipliers,
                test_baselines,
            ),
        ],
        how="vertical_relaxed",
    )
    predictions.write_parquet(output_dir / "step4_predictions.parquet")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=Path(
            "data/processed/"
            "player_week_features_2021-2022-2023-2024-2025.parquet"
        ),
        help="Leakage-safe player-week feature dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory for the model artifact, metrics, and predictions",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = train_and_evaluate(args.feature_path, args.output_dir)
    print(f"selected_model: {report['selected_model']['name']}")
    print(
        "validation_mae: "
        f"{report['validation']['selected_model_metrics']['overall']['mae']:.3f}"
    )
    print(
        "test_mae: "
        f"{report['test']['selected_model_metrics']['overall']['mae']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
