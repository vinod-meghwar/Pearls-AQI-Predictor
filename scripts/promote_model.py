import os
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load variables from .env file for local development
load_dotenv()

# CONFIGURATION
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
MODEL_NAME = "AQI_MultiOutput_Predictor"
ALIAS = "champion"  # This replaces the "Production" stage


def promote_model():
    client = MlflowClient()
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # 1. Get ALL versions to find the absolute newest
    all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not all_versions:
        print("No models found in registry.")
        return

    # Sort descending so index 0 is the highest version number
    all_versions.sort(key=lambda x: int(x.version), reverse=True)
    latest_ver = all_versions[0]

    # 2. Get current Champion model via Alias (instead of Stage)
    try:
        prod_ver = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    except Exception:
        prod_ver = None

    # 3. If newest is already the Champion, stop
    if prod_ver and latest_ver.version == prod_ver.version:
        print(f"Version {latest_ver.version} is already the {ALIAS}. No action needed.")
        return

    # 4. Fetch Metrics for newest
    latest_run = client.get_run(latest_ver.run_id)
    latest_r2 = latest_run.data.metrics.get("avg_r2", -1)

    # 5. Handle First Run (No Champion exists yet)
    if not prod_ver:
        print(f"No {ALIAS} found. Assigning Version {latest_ver.version} as {ALIAS}.")
        client.set_registered_model_alias(MODEL_NAME, ALIAS, latest_ver.version)
        return

    # 6. Fetch Metrics for current Champion
    prod_run = client.get_run(prod_ver.run_id)
    prod_r2 = prod_run.data.metrics.get("avg_r2", -1)

    # 7. Check Age for stale refresh
    prod_time = datetime.fromtimestamp(prod_ver.creation_timestamp / 1000, tz=timezone.utc)
    age_days = (datetime.now(timezone.utc) - prod_time).days

    print(f"--- Comparison Report ---")
    print(f"CURRENT {ALIAS}: v{prod_ver.version} | R2: {prod_r2:.4f} | Age: {age_days}d")
    print(f"CHALLENGER:   v{latest_ver.version} | R2: {latest_r2:.4f}")

    # 8. Decision Logic
    if latest_r2 > prod_r2 or age_days >= 3:
        reason = "Better Performance" if latest_r2 > prod_r2 else "Model is 3+ days old"
        print(f"RESULT: Promoting Version {latest_ver.version} to {ALIAS} because: {reason}.")

        # This modern call automatically removes the alias from the old version
        client.set_registered_model_alias(MODEL_NAME, ALIAS, latest_ver.version)
    else:
        print(f"RESULT: Current {ALIAS} remains superior. Challenger rejected.")


if __name__ == "__main__":
    promote_model()