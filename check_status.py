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
    
    print(f"✓ MongoDB Status:")
    print(f"  raw_data: {raw_count} documents")
    print(f"  feature_store: {feature_count} documents")
    
    if feature_count > 0:
        print(f"\n✓ Pipeline ready! Dashboard can now generate forecasts.")
    else:
        print(f"\n⏳ Feature engineering still running...")
    
except Exception as e:
    print(f"✗ Error: {e}")
