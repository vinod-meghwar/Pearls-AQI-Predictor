# AQI Predictor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pearls-aqi-predictor-app.streamlit.app/)

**Live app:** [pearls-aqi-predictor-app.streamlit.app](https://pearls-aqi-predictor-app.streamlit.app/)

An end-to-end air-quality forecasting platform for Hyderabad, Pakistan. The project collects hourly environmental data, builds time-series features, trains a multi-horizon model, tracks it with MLflow, and presents the latest AQI outlook in a Streamlit dashboard.

## What it does

- Collects weather and air-quality observations from Open-Meteo.
- Stores raw observations and engineered features in MongoDB Atlas.
- Forecasts AQI-related values for 24, 48, and 72 hours ahead.
- Compares LightGBM and Random Forest candidates using MAE, RMSE, and R2.
- Registers the winning model in DagsHub MLflow under the `champion` alias.
- Displays current conditions, trends, forecasts, health guidance, and downloadable reports.

## Architecture

```mermaid
%%{init: {"themeVariables": {"lineColor": "#64748b", "edgeLabelBackground": "#ffffff"}}}%%
flowchart TD

subgraph group_ingestion["Hourly data pipeline"]
  node_open_meteo{{"Open-Meteo<br/>external API"}}
  node_extract["Data extraction<br/>Python job<br/>[data_extraction.py]"]
  node_raw_data[("raw_data<br/>MongoDB collection")]
  node_features["Feature engineering<br/>Python job"]
  node_feature_store[("feature_store<br/>MongoDB collection")]
  node_mongo_atlas[("MongoDB Atlas<br/>data platform")]
end

subgraph group_ml["Training and promotion"]
  node_train["Model training<br/>Python job<br/>[model_train.py]"]
  node_mlflow[("DagsHub MLflow<br/>experiment registry")]
  node_registered_model["AQI MultiOutput Predictor<br/>registered model"]
  node_promote["Model promotion<br/>Python job<br/>[promote_model.py]"]
  node_champion["champion alias<br/>serving contract"]
end

subgraph group_operations["Scheduling and presentation"]
  node_pipeline_runner["Pipeline runner<br/>CLI orchestrator<br/>[pipeline_runner.py]"]
  node_github_actions["GitHub Actions<br/>scheduler"]
  node_hourly_workflow["Hourly workflow<br/>scheduled workflow<br/>[hourly_data.yml]"]
  node_daily_workflow["Daily workflow<br/>scheduled workflow<br/>[daily_model.yml]"]
  node_hourly_wrapper["Hourly wrapper<br/>automation script"]
  node_daily_wrapper["Daily wrapper<br/>automation script"]
  node_dashboard["Streamlit dashboard<br/>presentation runtime<br/>[app.py]"]
end

node_open_meteo -->|"hourly observations"| node_extract
node_extract -->|"writes"| node_raw_data
node_raw_data -->|"reads"| node_features
node_features -->|"writes"| node_feature_store
node_raw_data -.->|"hosted in"| node_mongo_atlas
node_feature_store -.->|"hosted in"| node_mongo_atlas
node_feature_store -->|"training data"| node_train
node_train -->|"logs metrics and model"| node_mlflow
node_mlflow -->|"registers"| node_registered_model
node_registered_model -->|"selected version"| node_promote
node_promote -->|"assigns alias"| node_champion
node_champion -->|"resolved for forecasts"| node_dashboard
node_mongo_atlas -->|"history and features"| node_dashboard
node_github_actions -->|"schedules"| node_hourly_workflow
node_github_actions -->|"schedules"| node_daily_workflow
node_hourly_workflow -->|"invokes"| node_hourly_wrapper
node_daily_workflow -->|"invokes"| node_daily_wrapper
node_hourly_wrapper -->|"hourly mode"| node_pipeline_runner
node_daily_wrapper -->|"daily mode"| node_pipeline_runner
node_pipeline_runner -->|"hourly"| node_extract
node_pipeline_runner -->|"hourly"| node_features
node_pipeline_runner -->|"daily"| node_train
node_pipeline_runner -->|"daily"| node_promote

click node_extract "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/scripts/data_extraction.py"
click node_features "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/scripts/feature_engineering.py"
click node_train "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/scripts/model_train.py"
click node_promote "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/scripts/promote_model.py"
click node_pipeline_runner "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/scripts/pipeline_runner.py"
click node_hourly_workflow "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/.github/workflows/hourly_data.yml"
click node_daily_workflow "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/.github/workflows/daily_model.yml"
click node_hourly_wrapper "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/automation_scripts/hourly_data_pipeline.py"
click node_daily_wrapper "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/automation_scripts/daily_model_pipeline.py"
click node_dashboard "https://github.com/vinod-meghwar/pearls-aqi-predictor/blob/main/app/app.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_open_meteo,node_extract,node_raw_data,node_features,node_feature_store,node_mongo_atlas toneBlue
class node_train,node_mlflow,node_registered_model,node_promote,node_champion toneAmber
class node_pipeline_runner,node_github_actions,node_hourly_workflow,node_daily_workflow,node_hourly_wrapper,node_daily_wrapper,node_dashboard toneMint

linkStyle default stroke:#64748b,stroke-width:1.8px
```

The default location is Hyderabad, Pakistan (`25.3548, 68.3585`).

## Dashboard

Live app: **[pearls-aqi-predictor-app.streamlit.app](https://pearls-aqi-predictor-app.streamlit.app/)**

The Streamlit app provides:

- Current AQI status and category.
- Three-day forecast outlook.
- Interactive historical and forecast trend charts.
- Health recommendations by AQI band.
- Historical environmental insights.
- Downloadable forecast data.
- Live model version and registry status.

### Dashboard Preview

[![Dashboard overview](https://github.com/vinod-meghwar/Pearls-AQI-Predictor/raw/main/assets/dashboard-overview.png)](https://pearls-aqi-predictor-app.streamlit.app/)

[![AQI trend](https://github.com/vinod-meghwar/Pearls-AQI-Predictor/raw/main/assets/aqi-trend.png)](https://pearls-aqi-predictor-app.streamlit.app/)

[![Data insights](https://github.com/vinod-meghwar/Pearls-AQI-Predictor/raw/main/assets/data-insights.png)](https://pearls-aqi-predictor-app.streamlit.app/)

[![Environmental overview](https://github.com/vinod-meghwar/Pearls-AQI-Predictor/raw/main/assets/environmental-overview.png)](https://pearls-aqi-predictor-app.streamlit.app/)

## Data and model flow

MongoDB contains two primary collections:

| Collection      | Purpose                                             |
| --------------- | --------------------------------------------------- |
| `raw_data`      | Hourly weather and air-quality observations.        |
| `feature_store` | Model-ready features and 24h, 48h, and 72h targets. |

The feature pipeline includes PM2.5 transformations, lag values, rolling statistics, pollutant interactions, wind vectors, cyclical time features, stagnation indicators, and weather changes.

The registered model is named `AQI_MultiOutput_Predictor`. The dashboard loads the version assigned to the `champion` alias.

## Project structure

```
app/app.py                         Streamlit dashboard
scripts/data_extraction.py         Open-Meteo ingestion
scripts/feature_engineering.py     Feature and target generation
scripts/model_train.py             Model training and registration
scripts/promote_model.py           Champion alias promotion
scripts/pipeline_runner.py         Hourly and daily pipeline runner
automation_scripts/                Scheduled pipeline entry points
.github/workflows/                 GitHub Actions automation
requirements.txt                   Local and dashboard dependencies
requirements-ci.txt                CI pipeline dependencies
```

## Local setup

### 1. Create an environment

Python 3.12 is used by GitHub Actions. A virtual environment is recommended locally:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the repository root. Never commit this file.

```
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/YOUR_DATABASE_NAME?retryWrites=true&w=majority&authSource=admin
DB_NAME=YOUR_DATABASE_NAME
MLFLOW_TRACKING_URI=https://dagshub.com/your-dagshub-username/your-repository.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your_dagshub_access_token
```

`MONGO_URI` uses MongoDB Atlas credentials. `MLFLOW_TRACKING_PASSWORD` uses a DagsHub access token. They are separate credentials. URL-encode special characters in the MongoDB password before placing it in the URI.

Replace `YOUR_DATABASE_NAME` with any database name you choose. Use the same name in both `MONGO_URI` and `DB_NAME`. The Atlas database user must have `readWrite` access to that database. The `.env` file is excluded by `.gitignore`.

### 3. Run the hourly pipeline

```
python scripts/pipeline_runner.py hourly
```

This runs data extraction followed by feature engineering.

### 4. Train and promote a model

```
python scripts/pipeline_runner.py daily
```

This trains the candidate models, logs metrics and artifacts to MLflow, registers the winner, and updates the `champion` alias.

### 5. Start the dashboard

```
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

Or skip local setup and use the hosted dashboard directly: **[pearls-aqi-predictor-app.streamlit.app](https://pearls-aqi-predictor-app.streamlit.app/)**

## GitHub Actions

The repository includes two scheduled workflows:

| Workflow             | Schedule                     | Purpose                                    |
| -------------------- | ----------------------------- | ------------------------------------------ |
| Hourly Data Pipeline | At 5 minutes past every hour | Collects data and updates features.        |
| Daily Model Training | Daily at midnight UTC        | Trains, registers, and promotes the model. |

Manual runs are available from the **Actions** tab with **Run workflow**.

Add these repository secrets under **Settings > Secrets and variables > Actions**:

```
MONGO_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_TRACKING_URI
```

Use the remote DagsHub MLflow URI shown above. The workflows validate required configuration before starting the pipeline.

## Troubleshooting

**MongoDB authentication failed**

- Confirm the Atlas username and password in `MONGO_URI`.
- Confirm the Atlas user has `readWrite` access.
- Use `authSource=admin` in the connection string.
- URL-encode special characters in the password.
- Restart Streamlit after changing `.env`.

**MLflow returns 403**

- Confirm `MLFLOW_TRACKING_URI` points to the DagsHub repository.
- Confirm the username matches the DagsHub account that owns or can write to the repository.
- Replace the password with a valid DagsHub access token that can write to the repository.

**The live model does not load**

- Confirm that `AQI_MultiOutput_Predictor` has a `champion` alias.
- Confirm `skops` is installed from `requirements.txt`.
- Check network access to DagsHub and restart the dashboard.

## Security

Never commit MongoDB passwords, DagsHub tokens, API keys, or `.env` files. Store CI credentials only in GitHub Actions repository secrets and rotate credentials immediately if they are exposed.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
