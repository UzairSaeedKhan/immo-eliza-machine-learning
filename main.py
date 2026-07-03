"""
Orchestrates the full ML pipeline:
    load → clean → split → preprocess → train → evaluate → predict
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np

from src.preprocess import structural_clean_data, preprocess_data
from src.train     import train_linear_regression, train_random_forest, train_xgboost
from src.evaluate  import evaluate_model
from src.predict import predict_from_json


def main():
    # Load & clean
    print("Loading data...")
    df = pd.read_csv("./data/raw/half_cleaned_properties.csv")
    # df = pd.read_csv("./data/raw/scraped_properties.csv")
    df = structural_clean_data(df)
    df.to_csv("./data/cleaned/properties_for_ml.csv", index=False)
    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # Preprocess
    X_train, y_train, scaler, feature_cols = preprocess_data(train_df, fit=True)
    X_test,  y_test, _, _ = preprocess_data(test_df, scaler=scaler, fit=False, feature_columns=feature_cols)

    # Save scaler
    joblib.dump(scaler, "./models/scaler.joblib")
    joblib.dump(feature_cols, "./models/feature_columns.joblib")
    # Train
    models = {
        "Linear Regression": train_linear_regression(X_train, y_train),
        "Random Forest": train_random_forest(X_train, y_train),
        "XGBoost": train_xgboost(X_train, y_train),
    }

    # after training, before evaluate loop:
    xgb_cv = np.mean(cross_val_score(models["XGBoost"], X_train, y_train, cv=5, scoring="r2"))

    # update evaluate calls:
    for name, model in models.items():
        cv = xgb_cv if name == "XGBoost" else None
        evaluate_model(model, X_train, y_train, X_test, y_test, name, cv_score=cv)

    print("\n═══ Predictions on Test Properties ═══")
    predict_from_json()

if __name__ == "__main__":
    main()