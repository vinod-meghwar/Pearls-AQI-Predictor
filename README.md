# Pearls AQI Predictor

**An end-to-end air-quality forecasting platform for Hyderabad, Pakistan.**

The system ingests hourly weather and air-quality data, engineers time-series features, trains and tracks a multi-horizon machine learning model, and serves 24h / 48h / 72h AQI forecasts through an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)
![MLflow](https://img.shields.io/badge/Tracking-MLflow-0194E2)
![MongoDB](https://img.shields.io/badge/Database-MongoDB%20Atlas-47A248)
![License](https://img.shields.io/badge/License-Unspecified-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Dashboard Preview](#dashboard-preview)
- [Data and Model Flow](#data-and-model-flow)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [1. Set Up the Environment](#1-set-up-the-environment)
  - [2. Configure Environment Variables](#2-configure-environment-variables)
  - [3. Run the Hourly Pipeline](#3-run-the-hourly-pipeline)
  - [4. Train and Promote a Model](#4-train-and-promote-a-model)
  - [5. Launch the Dashboard](#5-launch-the-dashboard)
- [Automation with GitHub Actions](#automation-with-github-actions)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [License](#license)

---

## Overview

Pearls AQI Predictor is a complete pipeline for forecasting air quality — from raw data collection to a production-style dashboard. It is built around Hyderabad, Pakistan (`25.3548, 68.3585`) and combines:

- **Automated data collection** from [Open-Meteo](https://open-meteo.com/)
- **Feature engineering** for time-series air-quality modeling
- **Model training and comparison** (LightGBM vs. Random Forest)
- **Experiment tracking and model registry** via MLflow on DagsHub
- **A Streamlit dashboard** for live forecasts, trends, and health guidance

## Key Features

- Collects hourly weather and air-quality observations from Open-Meteo
- Stores raw observations and engineered features in MongoDB Atlas
- Forecasts AQI-related values 24, 48, and 72 hours ahead
- Compares LightGBM and Random Forest candidates using MAE, RMSE, and R²
- Registers the winning model in DagsHub MLflow under a `champion` alias
- Displays current conditions, historical trends, forecasts, health guidance, and downloadable reports

## Architecture

```
Open-Meteo
    |
    v
MongoDB Atlas --> Feature Engineering --> Model Training
       |                                        |
       +--> Streamlit Dashboard <------ DagsHub MLflow
```

Data flows from Open-Meteo into MongoDB Atlas, is transformed into model-ready features, used to train and evaluate candidate models tracked in MLflow, and finally surfaced to end users through the Streamlit dashboard.

## Dashboard Preview

| Overview | AQI Trend |
|---|---|
| ![Dashboard overview](assets/dashboard-overview.png) | ![AQI trend](assets/aqi-trend.png) |

| Data Insights | Environmental Overview |
|---|---|
| ![Data insights](assets/data-insights.png) | ![Environmental overview](assets/environmental-overview.png) |

The dashboard provides:

- Current AQI status and category
- A three-day forecast outlook
- Interactive historical and forecast trend charts
- Health recommendations by AQI band
- Historical environmental insights
- Downloadable forecast data
- Live model version and registry status

## Data and Model Flow

MongoDB stores two primary collections:

| Collection | Purpose |
|---|---|
| `raw_data` | Hourly weather and air-quality observations |
| `feature_store` | Model-ready features and 24h, 48h, and 72h targets |

The feature pipeline generates PM2.5 transformations, lag values, rolling statistics, pollutant interactions, wind vectors, cyclical time features, stagnation indicators, and weather-change signals.

The registered model is named `AQI_MultiOutput_Predictor`. The dashboard always loads the version currently assigned to the `champion` alias in the MLflow model registry.

## Project Structure

```
app/app.py                         Streamlit dashboard
scripts/data_extraction.py         Open-Meteo data ingestion
scripts/feature_engineering.py     Feature and target generation
scripts/model_train.py             Model training and registration
scripts/promote_model.py           Champion alias promotion
scripts/pipeline_runner.py         Hourly and daily pipeline runner
automation_scripts/                Scheduled pipeline entry points
.github/workflows/                 GitHub Actions automation
requirements.txt                   Local and dashboard dependencies
requirements-ci.txt                CI pipeline dependencies
```

## Getting Started

### Prerequisites

- Python 3.12 (matches the version used in CI)
- A MongoDB Atlas cluster
- A DagsHub account with an MLflow-enabled repository

### 1. Set Up the Environment

A virtual environment is recommended:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the repository root (this file is already excluded via `.gitignore` — never commit it):

```env
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@CLUSTER.mongodb.net/YOUR_DATABASE_NAME?retryWrites=true&w=majority&authSource=admin
DB_NAME=YOUR_DATABASE_NAME
MLFLOW_TRACKING_URI=https://dagshub.com/your-dagshub-username/your-repository.mlflow
MLFLOW_TRACKING_USERNAME=your-dagshub-username
MLFLOW_TRACKING_PASSWORD=your_dagshub_access_token
```

**Notes:**
- `MONGO_URI` uses your MongoDB Atlas credentials; `MLFLOW_TRACKING_PASSWORD` uses a DagsHub access token — these are separate credentials.
- URL-encode any special characters in your MongoDB password before placing it in the connection string.
- Replace `YOUR_DATABASE_NAME` with a database name of your choice and use it consistently in both `MONGO_URI` and `DB_NAME`.
- The Atlas database user must have `readWrite` access to that database.

### 3. Run the Hourly Pipeline

```bash
python scripts/pipeline_runner.py hourly
```

Runs data extraction followed by feature engineering.

### 4. Train and Promote a Model

```bash
python scripts/pipeline_runner.py daily
```

Trains the candidate models, logs metrics and artifacts to MLflow, registers the winning model, and updates its `champion` alias.

### 5. Launch the Dashboard

```bash
streamlit run app/app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Automation with GitHub Actions

The repository ships with two scheduled workflows:

| Workflow | Schedule | Purpose |
|---|---|---|
| Hourly Data Pipeline | Every hour, at 5 minutes past | Collects data and updates features |
| Daily Model Training | Daily at midnight UTC | Trains, registers, and promotes the model |

Both workflows can also be triggered manually from the **Actions** tab using **Run workflow**.

Add the following repository secrets under **Settings → Secrets and variables → Actions**:

```
MONGO_URI
MLFLOW_TRACKING_USERNAME
MLFLOW_TRACKING_PASSWORD
MLFLOW_TRACKING_URI
```

Use the remote DagsHub MLflow URI shown above. Each workflow validates required configuration before the pipeline starts.

## Troubleshooting

<details>
<summary><strong>MongoDB authentication failed</strong></summary>

- Confirm the Atlas username and password in `MONGO_URI`.
- Confirm the Atlas user has `readWrite` access.
- Ensure `authSource=admin` is included in the connection string.
- URL-encode special characters in the password.
- Restart Streamlit after changing `.env`.
</details>

<details>
<summary><strong>MLflow returns 403</strong></summary>

- Confirm `MLFLOW_TRACKING_URI` points to the correct DagsHub repository.
- Confirm the username matches the DagsHub account that owns or can write to the repository.
- Replace the password with a valid DagsHub access token that has write access.
</details>

<details>
<summary><strong>The live model does not load</strong></summary>

- Confirm that `AQI_MultiOutput_Predictor` has a `champion` alias assigned.
- Confirm `skops` is installed from `requirements.txt`.
- Check network access to DagsHub and restart the dashboard.
</details>

## Security

Never commit MongoDB passwords, DagsHub tokens, API keys, or `.env` files. Store CI credentials only as GitHub Actions repository secrets, and rotate credentials immediately if they are ever exposed.

## License

No license has been selected for this repository yet.

---

<p align="center">Built for cleaner air insights in Hyderabad, Pakistan</p>
