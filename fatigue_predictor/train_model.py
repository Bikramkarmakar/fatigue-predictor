"""
train_model.py
--------------
Trains a Gradient Boosting regression model to predict next_day_fatigue_score
from three features:
  - bedtime_phone_minutes
  - screen_brightness_pct
  - caffeine_post_5pm_mg

Saves: model.pkl, scaler.pkl, and model_metrics.json
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "bedtime_screentime_sleep_debt.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")

# ── Feature / target names ────────────────────────────────────────────────────
FEATURES = ["bedtime_phone_minutes", "screen_brightness_pct", "caffeine_post_5pm_mg"]
TARGET = "next_day_fatigue_score"


def train():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded: {len(df)} rows")

    X = df[FEATURES].values
    y = df[TARGET].values

    # 2. Train / test split  (80 / 20, random_state fixed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Scale features
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 4. Train Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=5,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train_sc, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test_sc)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # 5-fold CV on full scaled dataset
    X_all_sc = scaler.transform(X)
    cv_scores = cross_val_score(model, X_all_sc, y, cv=5, scoring="r2")

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
        "model": "GradientBoostingRegressor",
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in zip(FEATURES, model.feature_importances_)
        },
    }

    print("\n-- Model Metrics ------------------------------------------")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  CV R2: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Feature importances: {metrics['feature_importances']}")

    # 6. Persist artefacts
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved: {MODEL_PATH}")
    print(f"Saved: {SCALER_PATH}")
    print(f"Saved: {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    train()
