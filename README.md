# AQI Predictor

An end-to-end air-quality forecasting platform for Hyderabad, Pakistan. The project collects hourly environmental data, builds time-series features, trains a multi-horizon model, tracks it with MLflow, and presents the latest AQI outlook in a Streamlit dashboard.

## What it does

- Collects weather and air-quality observations from Open-Meteo.
- Stores raw observations and engineered features in MongoDB Atlas.
- Forecasts AQI-related values for 24, 48, and 72 hours ahead.
- Compares LightGBM and Random Forest candidates using MAE, RMSE, and R2.
- Registers the winning model in DagsHub MLflow under the `champion` alias.
- Displays current conditions, trends, forecasts, health guidance, and downloadable reports.

## Architecture

```text
Open-Meteo
	|
	v
MongoDB Atlas --> Feature engineering --> Model training
	   |                                      |
	   +--> Streamlit dashboard <------ DagsHub MLflow
```

The default location is Hyderabad, Pakistan (`25.3548, 68.3585`).

## Dashboard

The Streamlit app provides:

- Current AQI status and category.
- Three-day forecast outlook.
- Interactive historical and forecast trend charts.
- Health recommendations by AQI band.
- Historical environmental insights.
- Downloadable forecast data.
- Live model version and registry status.

### Dashboard Preview

![Dashboard overview](assets/dashboard-overview.png)

![AQI trend](assets/aqi-trend.png)

![Data insights](assets/data-insights.png)

![Environmental overview](assets/environmental-overview.png)

## Data and model flow

MongoDB contains two primary collections:

| Collection | Purpose |
| --- | --- |
| `raw_data` | Hourly weather and air-quality observations. |
| `feature_store` | Model-ready features and 24h, 48h, and 72h targets. |

The feature pipeline includes PM2.5 transformations, lag values, rolling statistics, pollutant interactions, wind vectors, cyclical time features, stagnation indicators, and weather changes.

The registered model is named `AQI_MultiOutput_Predictor`. The dashboard loads the version assigned to the `champion` alias.

## Project structure

```text
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

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the repository root. Never commit this file.

```env
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/YOUR_DATABASE_NAME?retryWrites=true&w=majority&authSource=admin
DB_NAME=YOUR_DATABASE_NAME
MLFLOW_TRACKING_URI=https://dagshub.com/your-dagshub-username/your-repository.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your_dagshub_access_token
```

`MONGO_URI` uses MongoDB Atlas credentials. `MLFLOW_TRACKING_PASSWORD` uses a DagsHub access token. They are separate credentials. URL-encode special characters in the MongoDB password before placing it in the URI.

Replace `YOUR_DATABASE_NAME` with any database name you choose. Use the same name in both `MONGO_URI` and `DB_NAME`. The Atlas database user must have `readWrite` access to that database. The `.env` file is excluded by `.gitignore`.

### 3. Run the hourly pipeline

```powershell
python scripts/pipeline_runner.py hourly
```

This runs data extraction followed by feature engineering.

### 4. Train and promote a model

```powershell
python scripts/pipeline_runner.py daily
```

This trains the candidate models, logs metrics and artifacts to MLflow, registers the winner, and updates the `champion` alias.

### 5. Start the dashboard

```powershell
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

## GitHub Actions

The repository includes two scheduled workflows:

| Workflow | Schedule | Purpose |
| --- | --- | --- |
| Hourly Data Pipeline | At 5 minutes past every hour | Collects data and updates features. |
| Daily Model Training | Daily at midnight UTC | Trains, registers, and promotes the model. |

Manual runs are available from the **Actions** tab with **Run workflow**.

Add these repository secrets under **Settings > Secrets and variables > Actions**:

```text
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

No license has been selected for this repository yet.

