import os
import pandas as pd
import numpy as np
from pymongo import MongoClient, UpdateOne
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# SETUP
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["aqi_predictor"]
raw_col = db["raw_data"]
feature_col = db["feature_store"]


def run_feature_engineering():
    # Only fetch columns that actually help the model
    projection = {
        "_id": 0, "datetime": 1, "pm2_5": 1, "pm10": 1, "co": 1, "no2":1, "so2":1,
        "temperature": 1, "humidity": 1, "wind_speed": 1, "wind_dir": 1
    }

    is_empty = feature_col.count_documents({}) == 0

    if is_empty:
        print("Feature Store is empty. Full backfill...")
        cursor = raw_col.find({}, projection)
    else:
        print("Incremental update. Fetching 30-day window...")
        # CHANGED: 15 → 30 days (need 7d for rolling + 3d for targets + buffer)
        fetch_date = datetime.now(timezone.utc) - timedelta(days=30)
        cursor = raw_col.find({"datetime": {"$gte": fetch_date}}, projection)

    df = pd.DataFrame(list(cursor))
    if df.empty:
        print("No new data found.")
        return

    # Normalize Datetime
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(None).dt.tz_localize(timezone.utc)
    df = df.sort_values("datetime")

    # 1. Log Transformation (Stabilizes R2 against extreme pollution spikes)
    # Adding +1 to avoid log(0)
    df["pm25_log"] = np.log1p(df["pm2_5"])

    # 2. High-Impact Lags
    for lag in [1, 2, 3, 6, 12, 24, 48]:
        df[f"pm25_lag_{lag}h"] = df["pm25_log"].shift(lag)

    # 3. Rolling Statistics
    for window in [3, 6, 12, 24]:
        df[f"pm25_roll_mean_{window}h"] = df["pm25_log"].rolling(window).mean()
        df[f"pm25_roll_std_{window}h"] = df["pm25_log"].rolling(window).std()

    # 3.5 Pollutant interaction memory
    for col in ["pm10", "no2"]:
        df[f"{col}_lag_1h"] = df[col].shift(1)
        df[f"{col}_lag_6h"] = df[col].shift(6)

    # 4. Wind Vectors (Better than raw degrees for ML models)
    if "wind_speed" in df.columns and "wind_dir" in df.columns:
        df["wind_x"] = df["wind_speed"] * np.cos(np.deg2rad(df["wind_dir"]))
        df["wind_y"] = df["wind_speed"] * np.sin(np.deg2rad(df["wind_dir"]))

    # 5. Cyclical Time Features (Keeps both Sin and Cos for 24h cycles)
    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * (df["datetime"].dt.month - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["datetime"].dt.month - 1) / 12)

    # 5.5 Advanced Interaction Features
    # Stagnation: High values = high pollution + low wind (dangerous)
    df["stagnation_index"] = df["pm25_log"] / (df["wind_speed"] + 1)

    # Weekend Flag: Human activity cycles
    df["is_weekend"] = df["datetime"].dt.dayofweek.isin([5, 6]).astype(int)

    # Weather Tendency: Is it getting colder or warmer? (24h change)
    df["temp_diff_24h"] = df["temperature"].diff(24)
    df["hum_diff_24h"] = df["humidity"].diff(24)

    # 6. Target Generation (What we want to predict)
    df["target_h24"] = df["pm2_5"].shift(-24)
    df["target_h48"] = df["pm2_5"].shift(-48)
    df["target_h72"] = df["pm2_5"].shift(-72)

    target_cols = ["target_h24", "target_h48", "target_h72"]
    exclude_cols = target_cols + ["datetime"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Fill middle gaps using forward fill first
    df[feature_cols] = df[feature_cols].ffill()

    # Drop the starting rows where lags couldn't be calculated
    df.dropna(subset=["pm25_lag_48h"], inplace=True)

    # Final backfill for any remaining stray NaNs
    df[feature_cols] = df[feature_cols].bfill()

    if not is_empty:
        update_threshold = datetime.now(timezone.utc) - timedelta(days=14)
        df = df[df["datetime"] >= update_threshold]

    if df.empty:
        print("Feature store is already up to date.")
        return

    print(f"Syncing {len(df)} refined records to Feature Store...")
    records = df.to_dict("records")
    operations = [
        UpdateOne({"datetime": r["datetime"]}, {"$set": r}, upsert=True)
        for r in records
    ]

    if operations:
        result = feature_col.bulk_write(operations, ordered=False)
        print(f"Success! Upserts/Updates: {result.upserted_count + result.modified_count}")

if __name__ == "__main__":
    run_feature_engineering()