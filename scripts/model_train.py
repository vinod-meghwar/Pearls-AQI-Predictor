import os
import mlflow
import pandas as pd
import numpy as np
from pymongo import MongoClient
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Load variables from .env file for local development
load_dotenv()

# CONFIGURATION

def validate_mongo_uri(uri: str | None) -> str:
    if not uri:
        raise RuntimeError(
            "MONGO_URI is not set. Add it to .env locally or configure the GitHub Actions secret."
        )
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        raise RuntimeError(
            "MONGO_URI is malformed. It must begin with 'mongodb://' or 'mongodb+srv://'."
        )
    return uri


MONGO_URI = validate_mongo_uri(os.getenv("MONGO_URI"))
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "feature_store")

# DAGSHUB / MLFLOW SETUP
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")

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
        n_estimators=200, max_depth=6, learning_rate=0.08,
        subsample=0.9, colsample_bytree=0.9, objective="reg:squarederror",
        random_state=42, n_jobs=-1, tree_method="hist", verbosity=0
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train, y_train):
    print("Training MultiOutput LightGBM")
    base_model = LGBMRegressor(
        n_estimators=400, max_depth=8, num_leaves=31,
        learning_rate=0.05, subsample=0.8, random_state=42,
        n_jobs=-1, verbosity=-1
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    print("Training MultiOutput Random Forest")
    base_model = RandomForestRegressor(
        n_estimators=180, max_depth=12, min_samples_leaf=2,
        random_state=42, n_jobs=-1, max_features="sqrt"
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
    # 1. Point MLflow to the DagsHub URI
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("AQI-Prediction-Training")

    # Load and Split
    df = load_data()
    X, y, feature_cols = preprocess_data(df)

    # Use a simple temporal split for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Start MLflow Tracking
    run_name = f"Training_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):

        # 1. Train the reliable real model path (avoid XGBoost interruption in this environment)
        rf_model = train_random_forest(X_train, y_train)
        lgbm_model = train_lightgbm(X_train, y_train)
        xgb_model = None

        # 2. Measure both candidate models that are known to work reliably here
        rf_m = evaluate_model(rf_model, X_val, y_val)
        lgbm_m = evaluate_model(lgbm_model, X_val, y_val)

        # 3. Print Results Comparison Table for CI Logs
        print("\n" + "MODEL PERFORMANCE COMPARISON ".center(75, "="))
        print(f"{'Horizon':<10} | {'Metric':<5} | {'LightGBM':<10} | {'RandomForest':<12}")
        print("-" * 75)

        for h in ["24h", "48h", "72h"]:
            for m in ["MAE", "RMSE", "R2"]:
                print(f"{h if m == 'MAE' else '':<10} | {m:<5} | {lgbm_m[h][m]:<10} | {rf_m[h][m]:<12}")
            print("-" * 75)

        # 4. Determine Winner
        scores = {
            "LGBM": {"rmse": avg_metric(lgbm_m, "RMSE"), "mae": avg_metric(lgbm_m, "MAE"), "r2": avg_metric(lgbm_m, "R2")},
            "RF": {"rmse": avg_metric(rf_m, "RMSE"), "mae": avg_metric(rf_m, "MAE"), "r2": avg_metric(rf_m, "R2")}
        }

        best_name = sorted(scores.items(), key=lambda x: (-x[1]["r2"], x[1]["rmse"]))[0][0]
        best_model_obj = {"LGBM": lgbm_model, "RF": rf_model}[best_name]

        print(f"\nWinner: {best_name} | Avg R2: {scores[best_name]['r2']:.3f}")

        # 5. Log Metrics and Params to DagsHub
        mlflow.log_param("winning_model_type", best_name)
        mlflow.log_param("num_features", len(feature_cols))
        mlflow.log_metric("avg_rmse", scores[best_name]["rmse"])
        mlflow.log_metric("avg_mae", scores[best_name]["mae"])
        mlflow.log_metric("avg_r2", scores[best_name]["r2"])

        # 6. Retrain on full dataset
        # Important: Retrain so the cloud model has seen 100% of available data
        print(f"Retraining {best_name} on full dataset for production...")
        best_model_obj.fit(X, y)

        # 7. Register model to DagsHub Registry and set champion alias
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model_obj,
            name="model",
            registered_model_name="AQI_MultiOutput_Predictor"
        )

        client = mlflow.tracking.MlflowClient()
        latest_version = client.search_model_versions("name='AQI_MultiOutput_Predictor'", max_results=1)
        latest_version = latest_version[0] if latest_version else None
        if latest_version is not None:
            client.set_registered_model_alias("AQI_MultiOutput_Predictor", "champion", latest_version.version)
            print(f"Champion alias assigned to version {latest_version.version}.")

        print("\n" + "SUCCESS".center(30, "-"))
        print("Metrics & Parameters logged to MLflow Experiments.")
        print("Best model registered in DagsHub Model Registry and promoted to champion alias.")

if __name__ == "__main__":
    main()
