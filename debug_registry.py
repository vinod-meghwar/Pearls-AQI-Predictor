import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('MLFLOW_TRACKING_USERNAME', '')
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('MLFLOW_TRACKING_PASSWORD', '')
os.environ['MLFLOW_TRACKING_URI'] = os.getenv('MLFLOW_TRACKING_URI', '')

import mlflow
from mlflow.tracking import MlflowClient

uri = os.environ['MLFLOW_TRACKING_URI']
print('URI=' + uri)
print('USER=' + os.environ['MLFLOW_TRACKING_USERNAME'])
print('PASS_LEN=' + str(len(os.environ['MLFLOW_TRACKING_PASSWORD'])))

mlflow.set_tracking_uri(uri)
client = MlflowClient()

try:
    models = list(client.search_registered_models())
    print('REG_COUNT=' + str(len(models)))
    print('MODEL_NAMES=' + str([m.name for m in models[:10]]))
except Exception as e:
    print('REG_ERR=' + type(e).__name__ + '|' + str(e))

try:
    v = client.get_model_version_by_alias('AQI_MultiOutput_Predictor', 'champion')
    print('ALIAS_OK=' + str(v.version) + '|' + str(v.current_stage))
except Exception as e:
    print('ALIAS_ERR=' + type(e).__name__ + '|' + str(e))

Path('registry_status.txt').write_text('\n'.join([
    'URI=' + uri,
    'USER=' + os.environ['MLFLOW_TRACKING_USERNAME'],
    'PASS_LEN=' + str(len(os.environ['MLFLOW_TRACKING_PASSWORD'])),
    'REG_COUNT=' + str(len(list(client.search_registered_models()))) if 'client' in globals() else 'REG_COUNT=unknown',
]) + '\n', encoding='utf-8')
