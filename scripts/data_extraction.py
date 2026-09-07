import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import openmeteo_requests
import requests_cache
from retry_requests import retry
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# 1. SETUP
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError(
        "MONGO_URI is not set. Add it to .env locally or configure the GitHub Actions secret."
    )
client = MongoClient(MONGO_URI)
db = client["aqi_predictor"]
raw_collection = db["raw_data"]

# API Client Setup
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)


def fetch_raw_data(days=730):
    lat, lon = 25.3548, 68.3585  # Hyderabad, Pakistan
    # Use UTC for the API request boundaries
    now_utc_now = datetime.now(timezone.utc)
    end_date = now_utc_now.date()
    start_date = end_date - timedelta(days=days)

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "timezone": "auto"
    }

    # Fetch responses
    w_resp = openmeteo.weather_api("https://archive-api.open-meteo.com/v1/archive", params={**params,
                                                                                            "hourly": ["temperature_2m",
                                                                                                       "relative_humidity_2m",
                                                                                                       "windspeed_10m",
                                                                                                       "winddirection_10m"]})[
        0]
    a_resp = openmeteo.weather_api("https://air-quality-api.open-meteo.com/v1/air-quality", params={**params,
                                                                                                    "hourly": ["pm2_5",
                                                                                                               "pm10",
                                                                                                               "carbon_monoxide",
                                                                                                               "nitrogen_dioxide",
                                                                                                               "sulphur_dioxide",
                                                                                                               "ozone",
                                                                                                               "dust"]})[
        0]

    h_w = w_resp.Hourly()

    time_range = pd.date_range(
        start=pd.to_datetime(h_w.Time(), unit="s", utc=True),
        end=pd.to_datetime(h_w.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h_w.Interval()),
        inclusive="left"
    )

    df = pd.DataFrame({
        "datetime": time_range,
        "temperature": h_w.Variables(0).ValuesAsNumpy(),
        "humidity": h_w.Variables(1).ValuesAsNumpy(),
        "wind_speed": h_w.Variables(2).ValuesAsNumpy(),
        "wind_dir": h_w.Variables(3).ValuesAsNumpy(),
        "pm2_5": a_resp.Hourly().Variables(0).ValuesAsNumpy(),
        "pm10": a_resp.Hourly().Variables(1).ValuesAsNumpy(),
        "co": a_resp.Hourly().Variables(2).ValuesAsNumpy(),
        "no2": a_resp.Hourly().Variables(3).ValuesAsNumpy(),
        "so2": a_resp.Hourly().Variables(4).ValuesAsNumpy(),
        "o3": a_resp.Hourly().Variables(5).ValuesAsNumpy(),
        "dust": a_resp.Hourly().Variables(6).ValuesAsNumpy()
    })

    return df


if __name__ == "__main__":
    now_utc = datetime.now(timezone.utc)

    # Check if we already have data
    count = raw_collection.count_documents({})

    days_to_fetch = 730 if count == 0 else 3

    print(f"Database has {count} records. Fetching last {days_to_fetch} days...")
    raw_df = fetch_raw_data(days=days_to_fetch)

    df_for_mongo = raw_df.copy()

    print("Standardizing database timestamps...")
    existing_times = set()
    for dt in raw_collection.distinct("datetime"):
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        existing_times.add(dt.astimezone(timezone.utc).isoformat())

    # 2. Filter new records (Must be NOT in DB AND NOT in the future)
    records_to_insert = []
    for record in df_for_mongo.to_dict("records"):
        dt_obj = record["datetime"]
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)

        current_time_iso = dt_obj.isoformat()

        # LOGIC: Only add if it's new AND its timestamp has actually passed
        if current_time_iso not in existing_times and dt_obj <= now_utc:
            record["datetime"] = dt_obj
            records_to_insert.append(record)

    if records_to_insert:
        raw_collection.insert_many(records_to_insert)
        print(f"Success! Added {len(records_to_insert)} NEW records.")
    else:
        print("Everything is up to date. 0 records added.")

    # 3. Final Print in PKT
    last_record = raw_collection.find_one(sort=[("datetime", -1)])
    if last_record:
        last_dt = last_record['datetime']
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        # Convert UTC to PKT for the console log
        pkt_time = last_dt + timedelta(hours=5)
        print(f"The most recent data in the DB is from: {pkt_time.strftime('%Y-%m-%d %I:%M:%S %p')} (PKT)")
