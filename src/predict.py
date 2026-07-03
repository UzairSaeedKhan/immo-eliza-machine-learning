"""
Loads saved models and predicts prices for properties in test_prediction.json.
Edit test_prediction.json to test different property scenarios.
"""

import json
import numpy as np
import pandas as pd
import joblib

from src.preprocess import preprocess_data


MODELS_DIR        = "./models"
PREDICTIONS_FILE  = "./test_prediction.json"


def predict_from_json(json_path: str = PREDICTIONS_FILE):
    """
    Reads properties from a JSON file and predicts price using all saved models.
    Each property's '_description' field is used as a label — not passed to model.
    """
    # Load models and scaler
    models = {
        "Linear Regression": joblib.load(f"{MODELS_DIR}/linear_regression.joblib"),
        "Random Forest":     joblib.load(f"{MODELS_DIR}/random_forest.joblib"),
        "XGBoost":           joblib.load(f"{MODELS_DIR}/xgboost.joblib"),
    }
    scaler          = joblib.load(f"{MODELS_DIR}/scaler.joblib")
    feature_columns = joblib.load(f"{MODELS_DIR}/feature_columns.joblib")

    # Load properties
    with open(json_path, "r") as f:
        properties = json.load(f)

    print(f"\n{'═' * 55}")
    print(f"  Predictions for {len(properties)} properties")
    print(f"{'═' * 55}")

    for prop in properties:
        description = prop.pop("_description", "Unknown property")
        df = pd.DataFrame([prop])

        X, _, _, _ = preprocess_data(df, scaler=scaler, fit=False, feature_columns=feature_columns)

        print(f"\n📍 {description}")
        print(f"   {'Model':<22} {'Predicted Price':>15}")
        print(f"   {'-'*38}")
        for model_name, model in models.items():
            price = np.expm1(model.predict(X)[0])
            print(f"   {model_name:<22} €{price:>14,.0f}")

    print(f"\n{'═' * 55}")
