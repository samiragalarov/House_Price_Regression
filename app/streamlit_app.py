"""Streamlit interface for the house price model."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.predict import predict_price


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="centered")
st.title("House Price Predictor")
st.caption("Enter a small set of important housing features. Missing fields are imputed by the training pipeline.")


def collect_features() -> dict[str, Any]:
    col1, col2 = st.columns(2)

    with col1:
        features = {
            "OverallQual": st.slider("Overall Quality", min_value=1, max_value=10, value=6),
            "GrLivArea": st.number_input("Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500),
            "GarageCars": st.number_input("Garage Capacity", min_value=0, max_value=6, value=2),
            "GarageArea": st.number_input("Garage Area (sq ft)", min_value=0, max_value=2000, value=480),
            "TotalBsmtSF": st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=4000, value=900),
            "FullBath": st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2),
            "YearBuilt": st.number_input("Year Built", min_value=1872, max_value=2025, value=2000),
            "LotArea": st.number_input("Lot Area (sq ft)", min_value=1000, max_value=100000, value=9000),
        }

    with col2:
        features.update(
            {
                "Neighborhood": st.selectbox(
                    "Neighborhood",
                    ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst", "NridgHt", "Gilbert"],
                ),
                "HouseStyle": st.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"]),
                "KitchenQual": st.selectbox("Kitchen Quality", ["TA", "Gd", "Ex", "Fa"]),
                "ExterQual": st.selectbox("Exterior Quality", ["TA", "Gd", "Ex", "Fa"]),
                "BsmtQual": st.selectbox("Basement Quality", ["TA", "Gd", "Ex", "Fa", "Missing"]),
                "SaleCondition": st.selectbox("Sale Condition", ["Normal", "Partial", "Abnorml", "Family", "Alloca"]),
                "TotRmsAbvGrd": st.number_input("Rooms Above Grade", min_value=2, max_value=15, value=7),
                "Fireplaces": st.number_input("Fireplaces", min_value=0, max_value=4, value=1),
            }
        )

    return features


user_features = collect_features()

if st.button("Predict Price", type="primary"):
    try:
        prediction = predict_price(user_features)
        st.success(f"Estimated sale price: ${prediction:,.0f}")
    except FileNotFoundError:
        st.error("Model file not found. Train the model first with `python -m src.train`.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
