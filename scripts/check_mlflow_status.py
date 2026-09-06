import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', '')

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
client = MlflowClient()
models = list(client.search_registered_models())
lines = []
lines.append(f"REG_COUNT={len(models)}")
lines.append("MODEL_NAMES=" + str([m.name for m in models[:10]]))
try:
    v = client.get_model_version_by_alias('AQI_MultiOutput_Predictor', 'champion')
    lines.append(f"ALIAS_OK={v.version}|{v.current_stage}")
except Exception as e:
    lines.append(f"ALIAS_ERR={type(e).__name__}|{str(e)}")

out = Path('mlflow_status.txt')
out.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))

