# ==============================
# TensorFlow LSTM - TRAIN TEST
# PM2.5 24h / 48h / 72h Forecast
# ==============================
import os
import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from dotenv import load_dotenv

# Load variables from .env file for local development
load_dotenv()

# CONFIGURATION
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
COLLECTION = os.getenv("COLLECTION_NAME", "feature_store")

SEQUENCE_LENGTH = 72   # past 72 hours
TARGET_COLS = ["target_h24", "target_h48", "target_h72"]

FEATURE_COLS = [
    "pm25_log", "pm2_5",
    "pm25_roll_mean_24h", "pm25_roll_mean_72h", "pm25_roll_mean_168h",
    "pm25_roll_std_24h", "pm25_roll_std_72h", "pm25_roll_std_168h",
    "pm25_lag_1h",
    "month_sin", "month_cos",
    "temperature", "wind_x", "wind_y"
]

# LOAD DATA
def load_data():
    print("Loading Feature Store...")
    client = MongoClient(MONGO_URI)
    df = pd.DataFrame(list(client[DB_NAME][COLLECTION].find({}, {"_id": 0})))

    if df.empty:
        raise ValueError("Feature store is empty")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime")

    df = df[FEATURE_COLS + TARGET_COLS]
    df = df.dropna()

    print(f"Loaded {len(df)} clean rows")
    return df

# SEQUENCE BUILDER
def build_sequences(X, y, seq_len):
    X_seq, y_seq = [], []

    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


def main():
    df = load_data()

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLS])
    y = df[TARGET_COLS].values

    # Build sequences
    X_seq, y_seq = build_sequences(X_scaled, y, SEQUENCE_LENGTH)

    # Time-based split
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_seq[:split], y_seq[split:]

    print(f"Training samples: {X_train.shape}")
    print(f"Validation samples: {X_val.shape}")


    # LSTM MODEL
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(3)   # 24h, 48h, 72h
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # EVALUATION
    preds = model.predict(X_val)
    preds = np.maximum(preds, 0)

    horizons = ["24h", "48h", "72h"]
    print("\n" + "LSTM PERFORMANCE".center(40, "="))
    for i, h in enumerate(horizons):
        mae = mean_absolute_error(y_val[:, i], preds[:, i])
        r2 = r2_score(y_val[:, i], preds[:, i])
        print(f"{h} → MAE: {mae:.2f} | R2: {r2:.3f}")

if __name__ == "__main__":
    main()