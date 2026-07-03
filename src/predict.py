"""
Loads saved models and predicts prices for properties in test_prediction.json.
Edit test_prediction.json to test different property scenarios.
"""

import json
import numpy as np
import pandas as pd
import joblib
import warnings
# from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning)


MODELS_DIR        = "./models"
PREDICTIONS_FILE  = "./test_prediction.json"


def predict_from_json(json_path: str = PREDICTIONS_FILE):
    model = joblib.load(f"{MODELS_DIR}/xgboost.joblib")

    with open(json_path, "r") as f:
        properties = json.load(f)

    for prop in properties:
        description = prop.pop("_description", "Unknown property")
        df = pd.DataFrame([prop])

        print(f"\n {description}")
        price = np.expm1(model.predict(df)[0])
        print(f"   XGBoost prediction: €{price:>,.0f}")
