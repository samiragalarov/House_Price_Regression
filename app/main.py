"""FastAPI application for house price predictions."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from src.predict import predict_price


app = FastAPI(title="House Price Prediction API", version="1.0.0")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(features: dict[str, Any]) -> dict[str, float]:
    try:
        prediction = predict_price(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="Model file not found. Train the model first.") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
    return {"predicted_price": round(prediction, 2)}
