# House Price Prediction Application

Production-style refactor of the Kaggle House Prices regression project. The application keeps the original project direction: custom feature engineering plus Elastic Net regression on a log-transformed `SalePrice` target.

## Project Structure

```text
.
├── data/                  # Kaggle train/test CSV files
├── src/
│   ├── __init__.py
│   ├── preprocessing.py   # reusable preprocessing + feature engineering
│   ├── train.py           # training entry point
│   └── predict.py         # inference helpers + CLI
├── models/
│   └── model.pkl          # saved model artifact after training
├── app/
│   ├── main.py            # FastAPI app
│   └── streamlit_app.py   # Streamlit frontend
├── requirements.txt
└── README.md
```

## What The Pipeline Does

- adds engineered features from the original notebooks
- aligns raw input columns between training and inference
- imputes missing numerical values with the median
- imputes missing categorical values with `"Missing"`
- log-transforms highly skewed numeric predictors
- one-hot encodes categorical variables
- scales numeric features
- trains Elastic Net on `log1p(SalePrice)`

The saved artifact contains the full sklearn pipeline, so inference uses the exact same preprocessing logic as training.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train The Model

```bash
python -m src.train --data-path data/train.csv --model-path models/model.pkl
```

This command:

- loads raw training data
- runs cross-validated Elastic Net tuning
- trains the final model on the full dataset
- saves the artifact to `models/model.pkl`

## Run A Prediction From The CLI

```bash
python -m src.predict --input '{"OverallQual": 7, "GrLivArea": 1710, "GarageCars": 2, "TotalBsmtSF": 856}'
```

The script accepts either an inline JSON object or a path to a JSON file and returns a predicted sale price.

## Run The FastAPI Service

```bash
uvicorn app.main:app --reload
```

Prediction endpoint:

```http
POST /predict
Content-Type: application/json

{
  "OverallQual": 7,
  "GrLivArea": 1710,
  "GarageCars": 2,
  "GarageArea": 548,
  "TotalBsmtSF": 856,
  "FullBath": 2,
  "YearBuilt": 2003,
  "Neighborhood": "CollgCr"
}
```

Response:

```json
{
  "predicted_price": 208734.41
}
```

## Run The Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The Streamlit UI exposes a small set of the most influential features and relies on the trained pipeline to impute the rest.

## Notes

- On this macOS filesystem, `Data/` and `data/` resolve to the same directory, so the code uses `data/...` paths while remaining compatible with the existing folder.
- The notebook files are still present for reference, but training and inference should now go through the Python modules in `src/`.
