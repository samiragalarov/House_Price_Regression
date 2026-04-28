"""Train and persist the Elastic Net house price model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    TARGET_COLUMN,
    TrainingConfig,
    RawColumnAligner,
    FeatureEngineer,
    ColumnDropper,
    build_preprocessor,
)


def create_model_pipeline(config: TrainingConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("align_raw_columns", RawColumnAligner()),
            ("feature_engineering", FeatureEngineer()),
            ("drop_columns", ColumnDropper(list(config.columns_to_drop))),
            ("preprocessor", build_preprocessor()),
            ("model", ElasticNet(max_iter=20000, random_state=42)),
        ]
    )


def load_training_data(data_path: Path, config: TrainingConfig) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(data_path)
    X = data.drop(columns=[config.target_column])
    y = data[config.target_column]
    return X, y


def train_model(data_path: Path, model_path: Path) -> dict:
    config = TrainingConfig()
    X, y = load_training_data(data_path, config)
    y_log = np.log1p(y)

    X_train, X_valid, y_train_log, y_valid_log, y_train_actual, y_valid_actual = train_test_split(
        X,
        y_log,
        y,
        test_size=0.2,
        random_state=42,
    )

    pipeline = create_model_pipeline(config)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "model__alpha": [0.0005, 0.001, 0.005, 0.01, 0.05],
            "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    grid.fit(X_train, y_train_log)

    best_model = grid.best_estimator_
    valid_pred_log = best_model.predict(X_valid)
    valid_pred_actual = np.expm1(valid_pred_log)

    metrics = {
        "cv_rmse_log": float(-grid.best_score_),
        "holdout_rmse_log": float(np.sqrt(mean_squared_error(y_valid_log, valid_pred_log))),
        "holdout_rmse_price": float(np.sqrt(mean_squared_error(y_valid_actual, valid_pred_actual))),
        "holdout_r2_log": float(r2_score(y_valid_log, valid_pred_log)),
        "best_params": grid.best_params_,
    }

    final_model = create_model_pipeline(config)
    final_model.set_params(**grid.best_params_)
    final_model.fit(X, y_log)

    artifact = {
        "model": final_model,
        "target_transform": "log1p",
        "feature_columns": X.columns.tolist(),
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the house price Elastic Net model.")
    parser.add_argument("--data-path", default="data/train.csv", help="Path to the training CSV file.")
    parser.add_argument("--model-path", default="models/model.pkl", help="Where the trained model will be saved.")
    args = parser.parse_args()

    metrics = train_model(Path(args.data_path), Path(args.model_path))
    print("Training complete.")
    print(f"Model saved to {args.model_path}")
    print(f"Best params: {metrics['best_params']}")
    print(f"CV RMSE (log scale): {metrics['cv_rmse_log']:.5f}")
    print(f"Holdout RMSE (log scale): {metrics['holdout_rmse_log']:.5f}")
    print(f"Holdout RMSE (price): {metrics['holdout_rmse_price']:.2f}")
    print(f"Holdout R2 (log scale): {metrics['holdout_r2_log']:.4f}")


if __name__ == "__main__":
    main()
