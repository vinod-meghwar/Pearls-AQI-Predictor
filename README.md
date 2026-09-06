# AQI Predictor

This project builds and serves an air quality forecasting system for Hyderabad, Pakistan. It collects hourly weather and air-quality data from Open-Meteo, stores the raw records in MongoDB, engineers forecast features, trains a multi-output AQI model, and exposes a Streamlit dashboard that shows current AQI and a 3-day forecast.

## Overview

The system includes:

- Data ingestion from Open-Meteo
- MongoDB-backed storage for raw and engineered data
- Feature engineering for lag, rolling, cyclical, and meteorological signals
- Multi-output AQI forecasting for 24h, 48h, and 72h horizons
- MLflow tracking with a DagsHub registry and champion alias
- A Streamlit dashboard for monitoring and forecast presentation

## Project structure

- `app/app.py` - Streamlit dashboard for AQI monitoring and prediction
- `scripts/data_extraction.py` - pulls hourly weather and air-quality data
- `scripts/feature_engineering.py` - generates lag/rolling/time/weather features and targets
- `scripts/model_train.py` - trains and registers the forecasting model in MLflow
- `scripts/promote_model.py` - promotes the newest model version to the champion alias
- `scripts/pipeline_runner.py` - executes the hourly and daily pipeline steps
- `automation_scripts/hourly_data_pipeline.py` - runs the hourly data pipeline
- `automation_scripts/daily_model_pipeline.py` - runs the daily model pipeline
- `requirements.txt` - Python dependencies

## Data source

The project uses the Open-Meteo API for:

- Weather: temperature, humidity, wind speed, wind direction
- Air quality: PM2.5, PM10, CO, NO2, SO2, O3, dust

The default location is Hyderabad, Pakistan:

- Latitude: 25.3548
- Longitude: 68.3585

## MongoDB design

The application expects a MongoDB database named `aqi_predictor` with these collections:

- `raw_data` - raw hourly observations from Open-Meteo
- `feature_store` - engineered features with target columns for training

## MLflow and model registry

The code uses MLflow with a DagsHub tracking URI and a registered model named:

- `AQI_MultiOutput_Predictor`
- Alias: `champion`

The training script logs metrics such as:

- mean absolute error (MAE)
- root mean squared error (RMSE)
- R-squared (R2)

It chooses the strongest model between LightGBM and RandomForest and then registers the winning model.

## Features used for forecasting

The engineered feature set includes:

- PM2.5 log transform
- PM2.5 lag values for 1h, 2h, 3h, 6h, 12h, 24h, and 48h
- Rolling mean and standard deviation windows
- PM10 and NO2 lag features
- Wind vector features from speed and direction
- Hourly and monthly cyclic features
- Stagnation index
- Weekend indicator
- Temperature and humidity change features
- Target columns for 24h, 48h, and 72h prediction

## Dashboard

The Streamlit app is in `app/app.py` and provides:

- current AQI summary
- 3-day prediction outlook
- interactive AQI trend chart
- health recommendations by AQI band
- downloadable forecast report
- model status and registry metadata

To run the dashboard:

```bash
streamlit run app/app.py
```

## Pipeline execution

The project includes a command-line runner for the two main pipeline groups:

```bash
python scripts/pipeline_runner.py hourly
python scripts/pipeline_runner.py daily
```

These correspond to:

- `hourly`: `data_extraction.py`, `feature_engineering.py`
- `daily`: `model_train.py`, `promote_model.py`

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a `.env` file, including:

```env
MONGO_URI=your_mongodb_connection_string
DB_NAME=aqi_predictor
MLFLOW_TRACKING_URI=your_mlflow_tracking_uri
MLFLOW_TRACKING_USERNAME=your_dagshub_or_mlflow_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_or_mlflow_password
```

4. Run the data pipeline to populate MongoDB:

```bash
python scripts/pipeline_runner.py hourly
```

5. Train the model:

```bash
python scripts/pipeline_runner.py daily
```

6. Launch the dashboard:

```bash
streamlit run app/app.py
```

## Notes

- The app expects a working MongoDB connection and valid MLflow registry credentials.
- The dashboard falls back to a simple heuristic estimate if the champion model is not yet available.
- The project is designed for AQI forecasting and monitoring rather than production-grade deployment scaffolding.

## License

This project does not currently include a license file in the repository.
