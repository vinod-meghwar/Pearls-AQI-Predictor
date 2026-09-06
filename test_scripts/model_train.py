import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from pymongo import MongoClient
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATION
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "feature_store")

def load_data():
    print("Connecting to MongoDB Feature Store...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    cursor = collection.find({}, {"_id": 0})
    df = pd.DataFrame(list(cursor))

    if df.empty:
        raise ValueError("The feature_store is empty. Run the engineering script first!")

    print(f"Success! Loaded {len(df)} records.")
    return df

def preprocess_data(df):
    # 1. Chronological Sorting
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")
        df = df.drop(columns=["datetime"])

    # 2. Safety: Ensure numeric and drop NaNs
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # 3. Identify Targets and Features
    target_cols = ["target_h24", "target_h48", "target_h72"]
    feature_cols = [c for c in df.columns if c not in target_cols]

    X = df[feature_cols]
    y = df[target_cols]


    print(f"Training on {len(feature_cols)} features for {len(target_cols)} horizons.")
    return X, y, feature_cols

# MODEL TRAINING DEFS
def train_xgboost(X_train, y_train):
    print("Training MultiOutput XGBoost")
    base_model = XGBRegressor(
        n_estimators=1000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror",
        random_state=42, n_jobs=-1, tree_method="hist"
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    print("Training MultiOutput LightGBM")
    base_model = LGBMRegressor(
        n_estimators=1000, max_depth=10, num_leaves=31,
        learning_rate=0.05, subsample=0.8, random_state=42,
        n_jobs=-1, verbosity=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training MultiOutput Random Forest")
    base_model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=3,
        random_state=42, n_jobs=-1,max_features="sqrt"
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

# EVALUATION
def evaluate_model(model, X_val, y_val):
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)  # AQI cannot be negative

    horizons = ["24h", "48h", "72h"]
    metrics = {}
    for i, h in enumerate(horizons):
        mae = mean_absolute_error(y_val.iloc[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_val.iloc[:, i], preds[:, i]))
        r2 = r2_score(y_val.iloc[:, i], preds[:, i])
        metrics[h] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 3)
        }
    return metrics

def avg_metric(metrics, metric_name):
    return np.mean([metrics[h][metric_name] for h in metrics])

# MAIN EXECUTION
def main():
    # Load and Split
    df = load_data()
    X, y, feature_cols = preprocess_data(df)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # 1. Train the Models
    xgb_model = train_xgboost(X_train, y_train)
    lgbm_model = train_lightgbm(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # 2. Get Metrics
    xgb_m = evaluate_model(xgb_model, X_val, y_val)
    lgbm_m = evaluate_model(lgbm_model, X_val, y_val)
    rf_m = evaluate_model(rf_model, X_val, y_val)

    # 3. Print Results Comparison Table
    print("\n" + "MODEL PERFORMANCE COMPARISON ".center(75, "="))
    print(f"{'Horizon':<10} | {'Metric':<5} | {'XGBoost':<10} | {'LightGBM':<10} | {'RandomForest':<12}")
    print("-" * 75)

    for h in ["24h", "48h", "72h"]:
        for m in ["MAE", "RMSE", "R2"]:
            print(f"{h if m=='MAE' else '':<10} | {m:<5} | "
                  f"{xgb_m[h][m]:<10} | {lgbm_m[h][m]:<10} | {rf_m[h][m]:<12}")
        print("-" * 75)

    # 4. Determine Winner based on Average Scores
    scores = {
        "XGB": {
            "rmse": avg_metric(xgb_m, "RMSE"),
            "mae": avg_metric(xgb_m, "MAE"),
            "r2": avg_metric(xgb_m, "R2")
        },
        "LGBM": {
            "rmse": avg_metric(lgbm_m, "RMSE"),
            "mae": avg_metric(lgbm_m, "MAE"),
            "r2": avg_metric(lgbm_m, "R2")
        },
        "RF": {
            "rmse": avg_metric(rf_m, "RMSE"),
            "mae": avg_metric(rf_m, "MAE"),
            "r2": avg_metric(rf_m, "R2")
        }
    }

    best_name = sorted(
        scores.items(),
        key=lambda x: (-x[1]["r2"], x[1]["rmse"])
    )[0][0]

    best_model = {"XGB": xgb_model, "LGBM": lgbm_model, "RF": rf_model}[best_name]

    print(
        f"\nWinner: {best_name} | "
        f"Avg RMSE: {scores[best_name]['rmse']:.2f}, "
        f"Avg MAE: {scores[best_name]['mae']:.2f}, "
        f"Avg R2: {scores[best_name]['r2']:.3f}"
    )

    print("Retraining best model on full dataset...")
    best_model.fit(X, y)

    # 5. Export for Deployment
    # Path to project root (one level up from scripts/)
    project_root = Path(__file__).resolve().parent.parent

    # Path to models folder
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)

    # Full model path
    model_path = model_dir / "aqi_multi_output_model.pkl"

    # Save payload
    save_payload = {
        "model": best_model,
        "features": feature_cols,
        "model_type": best_name,
        "metrics": {
            "avg_rmse": scores[best_name]["rmse"],
            "avg_mae": scores[best_name]["mae"],
            "avg_r2": scores[best_name]["r2"]
        },
        "trained_at": str(pd.Timestamp.now())
    }

    MAX_MODEL_AGE_DAYS = 3
    MAX_ACCEPTABLE_DROP = 0.05  # allow noise

    should_replace = True

    if model_path.exists():
        current = joblib.load(model_path)
        old_date = pd.to_datetime(current.get("trained_at"))
        model_age_days = (pd.Timestamp.now() - old_date) / pd.Timedelta(days=1)

        old_r2 = current.get("metrics", {}).get("avg_r2", -1)
        new_r2 = scores[best_name]["r2"]

        # Always replace if model is too old
        if model_age_days >= MAX_MODEL_AGE_DAYS:
            print("Model is old. Replacing with newly trained model.")

        # If model is recent, apply safety gate
        else:
            # Check improvement / degradation
            r2_drop = old_r2 - new_r2

            if r2_drop > MAX_ACCEPTABLE_DROP:
                print(
                    f"New model degraded too much "
                    f"(old R²={old_r2:.3f}, new R²={new_r2:.3f}). Skipping replacement."
                )
                should_replace = False
            else:
                print("New model acceptable. Replacing existing model.")

    if should_replace:
        joblib.dump(save_payload, model_path, compress=3)
        print("Model saved successfully.")
    else:
        print("Existing model retained.")

if __name__ == "__main__":
    main()