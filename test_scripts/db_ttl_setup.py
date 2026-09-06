import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

# 1. SETUP
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["aqi_predictor"]
raw_col = db["raw_data"]
feature_col = db["feature_store"]

# 1. Define the query (identical to the shell version)
query = {"datetime": {"$type": "date"}}

# 2. Execute the query with a limit of 1
result = raw_col.find_one(query)

# 3. Check the result
if result:
    print("Success! Found a Date object:")
    print(result['datetime'])
    print(f"Type: {type(result['datetime'])}")
else:
    print("No records found with BSON Date type. They are likely still strings.")

# 1. Set the expiration using collMod (the correct way for Time-Series)
two_years_seconds = 60 * 60 * 24 * 365 * 2

try:
    db.command({
        "collMod": "raw_data",
        "expireAfterSeconds": two_years_seconds
    })
    print(f"SUCCESS: Set 'raw_data' to expire after {two_years_seconds} seconds.")
except Exception as e:
    print(f"Error setting expiration: {e}")

# 2. VERIFY the setting (Don't use list_indexes!)
# We check the collection options instead.
collections = db.list_collections(filter={"name": "raw_data"})
for col in collections:
    options = col.get("options", {})
    expire = options.get("expireAfterSeconds")

    if expire:
        print(f"VERIFIED: Data will delete after {expire} seconds (~{round(expire / (3600 * 24 * 365), 2)} years).")
    else:
        print("STILL MISSING: The expireAfterSeconds option is not set on this collection.")