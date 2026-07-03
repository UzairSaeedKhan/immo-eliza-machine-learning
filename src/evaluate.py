"""
Evaluates trained models with R², MSE, MAE and an overfitting check.
"""

import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, cv_score=None):
    """
    Prints R², MSE, MAE on test set and flags overfitting if
    train R² exceeds test R² by more than 0.1.
    Metrics computed in real € (inverse log-transform applied).
    """
    y_train_pred = np.expm1(model.predict(X_train))
    y_test_pred  = np.expm1(model.predict(X_test))
    y_train_true = np.expm1(y_train)
    y_test_true  = np.expm1(y_test)

    train_r2 = r2_score(y_train_true, y_train_pred)
    test_r2  = r2_score(y_test_true,  y_test_pred)
    test_mae = mean_absolute_error(y_test_true, y_test_pred)
    median_price = np.median(y_test_true)
    mae_pct      = (test_mae / median_price) * 100

    overfit = (train_r2 - test_r2) > 0.1

    print(f"\n {model_name} scores")
    print(f"  Train R²      : {train_r2:.4f}")
    print(f"  Test  R²      : {test_r2:.4f}")
    if cv_score is not None:
        print(f"  CV R²         : {cv_score:.4f}")
    print(f"  Test  MAE     : €{test_mae:,.0f}")
    print(f"  Median Price  : €{median_price:,.0f}")
    print(f"  Avg Error     : {mae_pct:.1f}% of median price")
    print(f"  Overfitting   : {'YES' if overfit else 'No'}")