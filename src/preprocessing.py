"""Reusable preprocessing components for house price prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "SalePrice"
ID_COLUMN = "Id"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator.div(denominator).replace([np.inf, -np.inf], np.nan).fillna(0.0)


class RawColumnAligner(BaseEstimator, TransformerMixin):
    """Adds missing raw columns at inference time and enforces training order."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "RawColumnAligner":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        for column in self.feature_names_in_:
            if column not in frame.columns:
                frame[column] = np.nan
        return frame[self.feature_names_in_]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the project's handcrafted features from the notebook workflow."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()

        if {"YrSold", "YearBuilt"}.issubset(frame.columns):
            frame["HouseAge"] = frame["YrSold"] - frame["YearBuilt"]

        if {"TotalBsmtSF", "1stFlrSF", "2ndFlrSF"}.issubset(frame.columns):
            frame["TotalSF"] = frame["TotalBsmtSF"] + frame["1stFlrSF"] + frame["2ndFlrSF"]

        if {"FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"}.issubset(frame.columns):
            frame["TotalBathrooms"] = (
                frame["FullBath"]
                + 0.5 * frame["HalfBath"]
                + frame["BsmtFullBath"]
                + 0.5 * frame["BsmtHalfBath"]
            )

        porch_columns = [col for col in ["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"] if col in frame]
        if porch_columns:
            frame["TotalPorchSF"] = frame[porch_columns].sum(axis=1)

        if {"BsmtFinSF1", "BsmtFinSF2"}.issubset(frame.columns):
            frame["FinishedBsmtSF"] = frame["BsmtFinSF1"] + frame["BsmtFinSF2"]

        if {"FinishedBsmtSF", "TotalBsmtSF"}.issubset(frame.columns):
            frame["FinishedBsmtRatio"] = _safe_ratio(frame["FinishedBsmtSF"], frame["TotalBsmtSF"])

        if {"GrLivArea", "LotArea"}.issubset(frame.columns):
            frame["LivingArea_to_LotArea"] = _safe_ratio(frame["GrLivArea"], frame["LotArea"])

        if {"TotalBsmtSF", "TotalSF"}.issubset(frame.columns):
            frame["Basement_to_TotalArea"] = _safe_ratio(frame["TotalBsmtSF"], frame["TotalSF"])

        if {"GarageArea", "GrLivArea"}.issubset(frame.columns):
            frame["Garage_to_LivingArea"] = _safe_ratio(frame["GarageArea"], frame["GrLivArea"])

        if {"YrSold", "YearRemodAdd"}.issubset(frame.columns):
            frame["YearsSinceRemodel"] = frame["YrSold"] - frame["YearRemodAdd"]

        if {"YearRemodAdd", "YearBuilt"}.issubset(frame.columns):
            frame["WasRemodeled"] = (frame["YearRemodAdd"] > frame["YearBuilt"]).astype(int)
            frame["IsRemodeled"] = (frame["YearRemodAdd"] != frame["YearBuilt"]).astype(int)

        if {"GarageYrBlt", "YrSold"}.issubset(frame.columns):
            garage_year = frame["GarageYrBlt"].fillna(0)
            frame["GarageAge"] = np.where(garage_year > 0, frame["YrSold"] - garage_year, 0)

        if "HouseAge" in frame.columns:
            frame["IsNewHouse"] = (frame["HouseAge"] <= 5).astype(int)

        if "2ndFlrSF" in frame.columns:
            frame["HasSecondFloor"] = (frame["2ndFlrSF"].fillna(0) > 0).astype(int)

        if "TotalBsmtSF" in frame.columns:
            frame["HasBasement"] = (frame["TotalBsmtSF"].fillna(0) > 0).astype(int)

        if "GarageArea" in frame.columns:
            frame["HasGarage"] = (frame["GarageArea"].fillna(0) > 0).astype(int)

        if "Fireplaces" in frame.columns:
            frame["HasFireplace"] = (frame["Fireplaces"].fillna(0) > 0).astype(int)

        if "PoolArea" in frame.columns:
            frame["HasPool"] = (frame["PoolArea"].fillna(0) > 0).astype(int)

        if porch_columns:
            frame["HasPorch"] = (frame[porch_columns].fillna(0).sum(axis=1) > 0).astype(int)

        if {"OverallQual", "GrLivArea"}.issubset(frame.columns):
            frame["Quality_x_Area"] = frame["OverallQual"] * frame["GrLivArea"]

        if {"OverallQual", "OverallCond"}.issubset(frame.columns):
            frame["Quality_x_Condition"] = frame["OverallQual"] * frame["OverallCond"]

        if {"TotalBathrooms", "BedroomAbvGr"}.issubset(frame.columns):
            frame["Baths_x_Bedrooms"] = frame["TotalBathrooms"] * frame["BedroomAbvGr"]

        if {"GarageCars", "GrLivArea"}.issubset(frame.columns):
            frame["GarageCars_x_Area"] = frame["GarageCars"] * frame["GrLivArea"]

        if "OverallQual" in frame.columns:
            frame["OverallQual_Squared"] = frame["OverallQual"] ** 2

        return frame


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops non-predictive or intentionally excluded columns."""

    def __init__(self, columns: list[str] | None = None) -> None:
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = pd.DataFrame(X).copy()
        return frame.drop(columns=self.columns, errors="ignore")


class SkewedLogTransformer(BaseEstimator, TransformerMixin):
    """Applies log1p to highly skewed non-negative numeric features."""

    def __init__(self, skew_threshold: float = 0.75) -> None:
        self.skew_threshold = skew_threshold

    def fit(self, X: np.ndarray, y: pd.Series | None = None) -> "SkewedLogTransformer":
        values = np.asarray(X, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        non_negative = np.nanmin(values, axis=0) >= 0
        skewness = pd.DataFrame(values).skew(axis=0, numeric_only=True).fillna(0.0).to_numpy()
        self.log_columns_ = np.where(non_negative & (np.abs(skewness) > self.skew_threshold))[0]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float).copy()
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if getattr(self, "log_columns_", None) is not None and len(self.log_columns_) > 0:
            values[:, self.log_columns_] = np.log1p(values[:, self.log_columns_])
        return values


@dataclass(frozen=True)
class TrainingConfig:
    target_column: str = TARGET_COLUMN
    id_column: str = ID_COLUMN
    columns_to_drop: tuple[str, ...] = (
        ID_COLUMN,
        "LivingArea_to_LotArea",
        "MiscVal",
        "BsmtFinSF2",
        "BsmtHalfBath",
        "YrSold",
        "PoolArea",
        "WasRemodeled",
    )


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log_skewed", SkewedLogTransformer(skew_threshold=0.75)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, make_column_selector(dtype_include=np.number)),
            ("cat", categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
        ],
        remainder="drop",
    )
