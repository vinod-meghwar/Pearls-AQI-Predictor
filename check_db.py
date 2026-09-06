import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    db = client["aqi_predictor"]
    
    raw_count = db["raw_data"].count_documents({})
    feature_count = db["feature_store"].count_documents({})
    
    print(f"✓ Connected to MongoDB")
    print(f"  raw_data: {raw_count} documents")
    print(f"  feature_store: {feature_count} documents")
    
    if raw_count == 0:
        print("\n⚠️  raw_data is empty! Need to run data extraction first.")
    elif feature_count == 0:
        print("\n✓ raw_data exists. Ready for feature engineering...")
    
except Exception as e:
    print(f"✗ Error: {e}")

