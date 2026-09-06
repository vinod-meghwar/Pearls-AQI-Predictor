import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
print(f"Connecting to MongoDB...")

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    db = client["aqi_predictor"]
    count = db["raw_data"].count_documents({})
    print(f"✓ MongoDB connected successfully")
    print(f"✓ raw_data collection has {count} documents")
except Exception as e:
    print(f"✗ MongoDB connection failed: {e}")

