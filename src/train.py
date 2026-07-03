"""
One function per model. Called from main.py with preprocessed data.
"""

import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


MODELS_DIR   = "models"
RANDOM_STATE = 42


def _save(model, filename):
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, filename))
    print(f"  Saved → models/{filename}")


def train_linear_regression(X_train, y_train):
    """Baseline linear model."""
    print("Training Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    _save(model, "linear_regression.joblib")
    return model


def train_random_forest(X_train, y_train):
    """Non-linear ensemble — robust to outliers and feature interactions."""
    print("Training Random Forest...")
    # model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1) # before
    model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=10,
    max_features=0.5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    )
    model.fit(X_train, y_train)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(f"Important values for random forest: {importances.sort_values(ascending=False).head(15)}")
    _save(model, "random_forest.joblib")
    return model


def train_xgboost(X_train, y_train):
    """XGBoost with RandomizedSearchCV hyperparameter tuning."""
    print("Training XGBoost with RandomizedSearchCV...")

    param_dist = {
        "n_estimators":      [200, 300, 500, 700],
        "learning_rate":     [0.01, 0.02, 0.05, 0.1],
        "max_depth":         [3, 4, 5, 6],
        "subsample":         [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9],
        "min_child_weight":  [1, 3, 5],
    }

    base_model = XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=30,          # tests 30 random combinations
        cv=5,               # 5-fold cross-validation
        scoring="r2",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV R²  : {search.best_score_:.4f}")

    model = search.best_estimator_
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print(f"Important values for xgboost: {importances.sort_values(ascending=False).head(15)}")
    _save(model, "xgboost.joblib")
    return model