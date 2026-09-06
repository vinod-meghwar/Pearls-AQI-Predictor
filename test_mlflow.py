import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

# Set MLflow credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))

print(f"MLflow URI: {os.getenv('MLFLOW_TRACKING_URI')}")
print(f"Testing connection...\n")

try:
    # Test registry connectivity
    client = mlflow.tracking.MlflowClient()
    
    # Try to list registered models
    print("✓ MLflow connection successful")
    
    models = client.search_registered_models()
    print(f"✓ Found {len(list(models))} registered models")
    
    # Check for our specific model
    try:
        model = client.get_registered_model("AQI_MultiOutput_Predictor")
        print(f"✓ Model 'AQI_MultiOutput_Predictor' exists")
        
        # Check for champion alias
        try:
            champion_version = client.get_model_version_by_alias("AQI_MultiOutput_Predictor", "champion")
            print(f"✓ Champion alias points to version {champion_version.version}")
            print(f"  Stage: {champion_version.current_stage}")
            print(f"  Status: {champion_version.status}")
        except Exception as e:
            print(f"✗ Champion alias error: {e}")
    except Exception as e:
        print(f"✗ Model not found: {e}")

except Exception as e:
    print(f"✗ Connection failed: {e}")
    print(f"\nReasons model registry is unavailable:")
    print("1. DagsHub/MLflow server is unreachable or slow")
    print("2. Authentication credentials are invalid")
    print("3. The model doesn't exist in the registry")
    print("4. Network connectivity issue")

