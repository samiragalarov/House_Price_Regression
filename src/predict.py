"""Inference helpers for the trained house price model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_artifact(model_path: str | Path = "models/model.pkl") -> dict:
    return joblib.load(model_path)


def predict_price(features: dict[str, Any], model_path: str | Path = "models/model.pkl") -> float:
    artifact = load_artifact(model_path)
    model = artifact["model"]
    frame = pd.DataFrame([features])
    prediction_log = float(model.predict(frame)[0])
    return float(np.expm1(prediction_log))


def _load_features_from_cli(raw_input: str) -> dict[str, Any]:
    path = Path(raw_input)
    if path.exists():
        return json.loads(path.read_text())
    return json.loads(raw_input)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a house price from raw JSON features.")
    parser.add_argument("--input", required=True, help="Inline JSON string or path to a JSON file.")
    parser.add_argument("--model-path", default="models/model.pkl", help="Path to the saved model artifact.")
    args = parser.parse_args()

    features = _load_features_from_cli(args.input)
    prediction = predict_price(features, model_path=args.model_path)
    print(json.dumps({"predicted_price": round(prediction, 2)}))


if __name__ == "__main__":
    main()
