import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load variables from .env file for local development
load_dotenv()

# 1. SETUP
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["aqi_predictor"]
raw_col = db["raw_data"]
feature_col = db["feature_store"]

# Get the oldest record
oldest = raw_col.find_one(sort=[("datetime", 1)])

if oldest:
    print(f"Oldest record currently: {oldest['datetime']}")

    # Calculate how many days old it is
    from datetime import datetime, timezone

    diff = datetime.now(timezone.utc) - oldest['datetime'].replace(tzinfo=timezone.utc)
    print(f"This record is {diff.days} days old.")
else:
    print("Collection is empty.")

# Check how the time-series was defined
col_info = db.command("listCollections", filter={"name": "raw_data"})
options = col_info['cursor']['firstBatch'][0].get('options', {})

print("Time-Series Options:", options.get('timeseries'))
print("Expire After Seconds:", options.get('expireAfterSeconds'))