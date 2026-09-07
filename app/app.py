import os
import streamlit as st
import mlflow
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pymongo import MongoClient
from dotenv import load_dotenv
import warnings
import io
import threading
from urllib.parse import urlparse

warnings.filterwarnings('ignore')

# Load Environment Variables
load_dotenv()

# CONFIGURATION
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")

MODEL_NAME = "AQI_MultiOutput_Predictor"
ALIAS = "champion"

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME", "aqi_predictor")
RAW_COLLECTION = "raw_data"
FEATURE_COLLECTION = "feature_store"
MODEL_LOAD_TIMEOUT_SECONDS = int(os.getenv("MODEL_LOAD_TIMEOUT_SECONDS", "180"))


def validate_mongo_uri(uri):
    if not uri:
        raise RuntimeError(
            "MONGO_URI is not configured. Add the MongoDB Atlas connection string to the project .env file."
        )

    parsed_uri = urlparse(uri)
    if parsed_uri.scheme not in ("mongodb", "mongodb+srv") or not parsed_uri.hostname:
        raise RuntimeError(
            "MONGO_URI is malformed. It must be a valid mongodb:// or mongodb+srv:// connection string."
        )

    if not parsed_uri.username or parsed_uri.password is None:
        raise RuntimeError(
            "MONGO_URI must include the MongoDB Atlas username and password."
        )

    return uri


try:
    MONGO_URI = validate_mongo_uri(MONGO_URI)
except RuntimeError as error:
    MONGO_URI_ERROR = str(error)
else:
    MONGO_URI_ERROR = None
# MODERN CSS STYLING
# PAGE SETTINGS
st.set_page_config(
    page_title="AQI Forecast Dashboard",
    layout="wide",
    page_icon="◆",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #102a43;
        --ink-soft: #486581;
        --canvas: #f4f7f5;
        --paper: #ffffff;
        --line: #d9e2ec;
        --teal: #087f8c;
        --teal-dark: #05616b;
        --mint: #d9f3ee;
        --amber: #f0b429;
        --amber-soft: #fff3c4;
        --shadow: 0 16px 40px rgba(16, 42, 67, 0.08);
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(circle at 88% 4%, rgba(8, 127, 140, 0.09), transparent 28rem),
            linear-gradient(180deg, #f8fbf9 0%, var(--canvas) 48%, #eef4f2 100%);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1420px; padding: 2.5rem 3.5rem 4rem; }


        background: linear-gradient(115deg, #102a43 0%, #164e63 62%, #087f8c 100%);
        padding: 2.25rem 2.5rem;
        border-radius: 18px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        text-align: left;
        margin-bottom: 2.25rem;
        box-shadow: 0 18px 45px rgba(16, 42, 67, 0.18);
        position: relative;
        overflow: hidden;
    }

    .main-header::after {
        content: 'AQI / LIVE';
        position: absolute;
        right: 2rem;
        top: 2rem;
        color: rgba(255,255,255,0.7);
        font: 600 0.72rem 'Space Grotesk', sans-serif;
        letter-spacing: 0.12em;
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3.35rem);
        color: white;
        letter-spacing: 0;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        margin: 0.7rem 0 0;
        font-size: 1rem;
        color: #c8f1ec;
    }

    h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
        letter-spacing: 0;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--ink-soft);
        margin: 0;
        font-size: 2.5rem;
    }
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.15rem !important;
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: var(--ink-soft);
        font-size: 0.72rem !important;

    /* Metric cards - Equal sizing and theme support */
        letter-spacing: 0.09em;
        font-size: 2rem !important;
        font-weight: 700 !important;
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.86);
        padding: 1.35rem 1.25rem;
        border-radius: 14px;
        box-shadow: var(--shadow);
        border: 1px solid rgba(217, 226, 236, 0.9);
        border-top: 3px solid var(--teal);
        min-height: 122px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;

    /* Dark theme metric cards */
        box-shadow: 0 20px 42px rgba(16, 42, 67, 0.13);
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        }
        background: var(--teal);

    div[data-testid="metric-container"]:hover {
        padding: 0.72rem 1.4rem;
        border-radius: 10px;
    }
        font-size: 0.9rem;
        transition: all 0.2s ease;
        box-shadow: 0 8px 18px rgba(8, 127, 140, 0.2);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        background: var(--teal-dark);
        transform: translateY(-1px);
        box-shadow: 0 12px 24px rgba(8, 127, 140, 0.26);
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.3);
        gap: 0;

        border-bottom: 1px solid var(--line);
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
        height: 44px;
        background: transparent;
        border-radius: 0;
        padding: 0 1.1rem;
        color: var(--ink-soft);
        font-weight: 600;
        border-bottom: 3px solid transparent;
        transition: color 0.2s, border-color 0.2s;

    .stTabs [data-baseweb="tab"] {

        color: var(--teal);
    @media (prefers-color-scheme: dark) {
        .stTabs [data-baseweb="tab"] {
            background-color: #2d3748;
        background: transparent;
        border-bottom-color: var(--teal);
        color: var(--teal) !important;
    }

        border-color: #667eea;
        background: var(--paper);
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: var(--shadow);
        border-top: 1px solid rgba(217, 226, 236, 0.8);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);

    [data-testid="stSidebar"] {
        background: #102a43;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #d9f3ee !important; }

    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background: rgba(255,255,255,0.09);
        border: 1px solid rgba(217,243,238,0.18);
    }

    [data-testid="stSidebar"] hr { border-color: rgba(217,243,238,0.16); }

    @media (max-width: 760px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        .main-header { padding: 1.6rem; }
        .main-header::after { display: none; }
        .main-header h1 { font-size: 2rem; }
    }
    }

    /* Dark theme alert cards */
    @media (prefers-color-scheme: dark) {
        .alert-card {
            background: #2d3748;
            color: #e2e8f0;
        }
    }

    /* Info boxes */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        border-left-width: 5px;
    }

    /* Download button styling */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# CACHED MODEL LOADING WITH AUTO-UPDATE
@st.cache_resource(show_spinner=False)
def load_champion_model():
    """Loads the model from MLflow Registry with error handling and timeout."""
    result = {"model": None, "version": None, "error": "Timeout loading model"}
    
    def _load_model():
        try:
            if not os.environ.get("MLFLOW_TRACKING_URI"):
                raise ValueError("MLflow tracking URI not configured")

            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
            model_uri = f"models:/{MODEL_NAME}@{ALIAS}"

            model = mlflow.pyfunc.load_model(model_uri)
            client = mlflow.tracking.MlflowClient()
            model_ver = client.get_model_version_by_alias(MODEL_NAME, ALIAS)

            result["model"] = model
            result["version"] = model_ver.version
            result["error"] = None
        except Exception as e:
            result["error"] = str(e)
    
    # Give the registry/artifact fetch enough time to complete over DagsHub.
    thread = threading.Thread(target=_load_model, daemon=True)
    thread.start()
    thread.join(timeout=MODEL_LOAD_TIMEOUT_SECONDS)

    if thread.is_alive():
        result["error"] = (
            f"Model loading timed out after {MODEL_LOAD_TIMEOUT_SECONDS} seconds "
            "while fetching the champion artifact from MLflow."
        )
    
    return result["model"], result["version"], result["error"]


def check_for_model_updates(current_version):
    """
    Check if a new champion model version is available in the registry.
    Returns (has_update, new_version, error)
    """
    try:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        client = mlflow.tracking.MlflowClient()

        # Get the latest champion model version
        latest_model = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
        latest_version = latest_model.version

        # Check if there's a new version
        if latest_version != current_version:
            return True, latest_version, None

        return False, current_version, None

    except Exception as e:
        return False, current_version, str(e)


# UTILITY FUNCTIONS
def calculate_aqi(pm25):
    """Calculate AQI from PM2.5 concentration using EPA formula."""
    if pm25 < 0:
        return 0
    if pm25 <= 12.0:
        return round(((50 - 0) / (12.0 - 0)) * (pm25 - 0) + 0)
    elif pm25 <= 35.4:
        return round(((100 - 51) / (35.4 - 12.1)) * (pm25 - 12.1) + 51)
    elif pm25 <= 55.4:
        return round(((150 - 101) / (55.4 - 35.5)) * (pm25 - 35.5) + 101)
    elif pm25 <= 150.4:
        return round(((200 - 151) / (150.4 - 55.5)) * (pm25 - 55.5) + 151)
    elif pm25 <= 250.4:
        return round(((300 - 201) / (250.4 - 150.5)) * (pm25 - 150.5) + 201)
    else:
        return 500


def get_aqi_info(aqi_val):
    """Get AQI category, color, and health recommendation."""
    if aqi_val <= 50:
        return ("Good", "#00e400", "Air quality is satisfactory. Ideal for all outdoor activities!")
    elif aqi_val <= 100:
        return ("Moderate", "#9a6700",
                "Air quality is acceptable. Sensitive groups should limit prolonged outdoor exertion.")
    elif aqi_val <= 150:
        return ("Unhealthy for Sensitive Groups", "#ff7e00",
                "Members of sensitive groups may experience health effects. Wear a mask if needed.")
    elif aqi_val <= 200:
        return (
            "Unhealthy", "#ff0000", "Everyone may experience health effects. Reduce prolonged outdoor activities.")
    elif aqi_val <= 300:
        return ("Very Unhealthy", "#8f3f97", "Health alert: Everyone should avoid all outdoor physical activity.")
    else:
        return ("Hazardous", "#7e0023", "Emergency conditions. Stay indoors with air purification systems.")


def create_aqi_chart_plotly(plot_dates, aqi_values, types):
    """An interactive AQI visualization chart using Plotly."""

    # Convert datetime objects to ensure compatibility
    plot_dates = [pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in plot_dates]

    # Create figure with better styling
    fig = go.Figure()

    # Add AQI category background bands with improved styling
    aqi_bands = [
        (0, 50, '#00e676', 'Good'),  # Bright Green
        (51, 100, '#ffd54f', 'Moderate'),  # Yellow
        (101, 150, '#ff9800', 'Unhealthy (Sensitive)'),  # Orange
        (151, 200, '#f44336', 'Unhealthy'),  # Red
        (201, 300, '#9c27b0', 'Very Unhealthy'),  # Purple
        (301, 500, '#6d1b7b', 'Hazardous')  # Dark Purple
    ]

    for low, high, color, label in aqi_bands:
        fig.add_hrect(
            y0=low, y1=high,
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation=dict(
                font_size=11,
                font_color=color,
                font_family='Arial, sans-serif'
            )
        )

    # Add vertical line separating observed and predicted
    separator_idx = types.index("Predicted") if "Predicted" in types else len(types)
    if separator_idx < len(plot_dates):
        # Use shapes instead of add_vline to avoid datetime arithmetic issues
        fig.add_shape(
            type="line",
            x0=plot_dates[separator_idx],
            x1=plot_dates[separator_idx],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="#78909c", width=3, dash="dash"),
            opacity=0.6
        )
        # Add annotation separately
        fig.add_annotation(
            x=plot_dates[separator_idx],
            y=1,
            yref="paper",
            text="◆ Forecast Starts",
            showarrow=False,
            yshift=15,
            font=dict(size=12, color="#455a64", family='Arial, sans-serif', weight='bold'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#78909c",
            borderwidth=2,
            borderpad=4
        )

    # Split data into observed and predicted
    observed_dates = [d for d, t in zip(plot_dates, types) if t == "Observed"]
    observed_aqi = [a for a, t in zip(aqi_values, types) if t == "Observed"]
    predicted_dates = [d for d, t in zip(plot_dates, types) if t == "Predicted"]
    predicted_aqi = [a for a, t in zip(aqi_values, types) if t == "Predicted"]

    # Add main trend line (observed) with gradient effect
    if observed_dates:
        fig.add_trace(go.Scatter(
            x=observed_dates,
            y=observed_aqi,
            mode='lines+markers',
            name='Observed',
            line=dict(color='#1e88e5', width=4, shape='spline'),
            marker=dict(
                size=14,
                color=[get_aqi_info(a)[1] for a in observed_aqi],
                line=dict(color='white', width=3),
                symbol='circle'
            ),
            fill='tozeroy',
            fillcolor='rgba(30, 136, 229, 0.08)',
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>AQI: <b>%{y}</b><br>Category: %{text}<extra></extra>',
            text=[get_aqi_info(a)[0] for a in observed_aqi]
        ))

    # Add predicted trend line with different styling
    if predicted_dates:
        # Connect last observed to first predicted
        connect_dates = [observed_dates[-1], predicted_dates[0]] if observed_dates else predicted_dates
        connect_aqi = [observed_aqi[-1], predicted_aqi[0]] if observed_aqi else predicted_aqi

        fig.add_trace(go.Scatter(
            x=connect_dates + predicted_dates[1:],
            y=connect_aqi + predicted_aqi[1:],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#7b1fa2', width=4, dash='dot', shape='spline'),
            marker=dict(
                size=16,
                symbol='diamond',
                color=[get_aqi_info(a)[1] for a in connect_aqi + predicted_aqi[1:]],
                line=dict(color='white', width=3)
            ),
            fill='tozeroy',
            fillcolor='rgba(123, 31, 162, 0.08)',
            hovertemplate='<b>Date: %{x|%b %d, %Y}</b><br>Forecast AQI: <b>%{y}</b><br>Category: %{text}<extra></extra>',
            text=[get_aqi_info(a)[0] for a in connect_aqi + predicted_aqi[1:]]
        ))

    # Update layout with modern styling
    fig.update_layout(
        title={
            'text': 'Air Quality Index - Historical & Forecast Trend',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#1a237e', 'family': 'Arial, sans-serif', 'weight': 'bold'}
        },
        xaxis_title='Date',
        yaxis_title='AQI Index',
        hovermode='x unified',
        plot_bgcolor='#fafafa',
        paper_bgcolor='white',
        height=550,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#90a4ae",
            borderwidth=2,
            font=dict(size=12, family='Arial, sans-serif')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.06)',
            tickformat='%b %d\n%a',
            tickfont=dict(size=11, color='#37474f', family='Arial, sans-serif'),
            linecolor='#cfd8dc',
            linewidth=2
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            range=[0, max(aqi_values) + 50],
            tickfont=dict(size=11, color='#37474f', family='Arial, sans-serif'),
            linecolor='#cfd8dc',
            linewidth=2,
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.1)',
            zerolinewidth=2
        ),
        font=dict(family="Arial, sans-serif", size=12, color='#263238')
    )

    return fig


def create_historical_overview_chart(hist_df):
    """Create multi-parameter historical overview chart."""

    # Map column names from database to display names
    column_mapping = {
        'temperature': 'temp',
        'humidity': 'rh',
        'wind_speed': 'ws',
        'pressure': 'pres'
    }

    # Rename columns if they exist in old format
    for old_name, new_name in column_mapping.items():
        if old_name in hist_df.columns and new_name not in hist_df.columns:
            hist_df[new_name] = hist_df[old_name]

    # Create subplots with better styling - 2x2 grid for 4 parameters
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PM2.5 Concentration', 'PM10 Concentration',
                        'Temperature', 'Humidity'),
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )

    # Define parameters with modern, vibrant colors and their RGBA equivalents
    params = [
        ('pm2_5', 'PM2.5 (μg/m³)', '#FF6B6B', 'rgba(255, 107, 107, 0.15)', 1, 1),  # Coral Red
        ('pm10', 'PM10 (μg/m³)', '#4ECDC4', 'rgba(78, 205, 196, 0.15)', 1, 2),  # Turquoise
        ('temp', 'Temperature (°C)', '#45B7D1', 'rgba(69, 183, 209, 0.15)', 2, 1),  # Sky Blue
        ('rh', 'Humidity (%)', '#96CEB4', 'rgba(150, 206, 180, 0.15)', 2, 2)  # Mint Green
    ]

    for param, label, color, fill_color, row, col in params:
        if param in hist_df.columns:
            # Create gradient effect with area fill
            fig.add_trace(
                go.Scatter(
                    x=hist_df['datetime'],
                    y=hist_df[param],
                    mode='lines',
                    name=label,
                    line=dict(color=color, width=3, shape='spline'),
                    fill='tozeroy',
                    fillcolor=fill_color,
                    showlegend=False,
                    hovertemplate='<b>%{x|%b %d, %H:%M}</b><br>' + f'{label}: %{{y:.2f}}<extra></extra>'
                ),
                row=row, col=col
            )

            # Add subtle trend line
            if len(hist_df) > 10:
                # Calculate simple moving average for trend
                window = min(24, len(hist_df) // 4)
                trend = hist_df[param].rolling(window=window, center=True).mean()

                fig.add_trace(
                    go.Scatter(
                        x=hist_df['datetime'],
                        y=trend,
                        mode='lines',
                        line=dict(color=color, width=2, dash='dot'),
                        showlegend=False,
                        hovertemplate='<b>Trend</b><br>%{y:.2f}<extra></extra>',
                        opacity=0.6
                    ),
                    row=row, col=col
                )

    # layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title={
            'text': "Historical Environmental Parameters (Past 7 Days)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': '#2c3e50', 'family': 'Arial, sans-serif', 'weight': 'bold'}
        },
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white',
        hovermode='x unified',
        font=dict(family="Arial, sans-serif", size=12, color='#2c3e50')
    )

    # subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=14, color='#34495e', family='Arial, sans-serif', weight='bold')

    # axes with better styling
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.05)',
        tickfont=dict(size=10, color='#5a6c7d'),
        linecolor='#cbd5e0',
        linewidth=1
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.08)',
        tickfont=dict(size=10, color='#5a6c7d'),
        linecolor='#cbd5e0',
        linewidth=1,
        zeroline=True,
        zerolinecolor='rgba(0,0,0,0.1)',
        zerolinewidth=1
    )

    return fig


def create_download_dataframe(plot_dates, aqi_values, types, forecast_pm25=None):
    """Create comprehensive dataframe for download."""
    # Convert to pandas Timestamp if not already
    plot_dates_ts = [pd.Timestamp(d) if not isinstance(d, pd.Timestamp) else d for d in plot_dates]

    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in plot_dates_ts],
        'Day': [d.strftime('%A') for d in plot_dates_ts],
        'Time': [d.strftime('%H:%M:%S') for d in plot_dates_ts],
        'AQI': aqi_values,
        'Category': [get_aqi_info(a)[0] for a in aqi_values],
        'Type': types,
        'Health_Recommendation': [get_aqi_info(a)[2] for a in aqi_values]
    })

    # Add PM2.5 values for predicted days if available
    if forecast_pm25 is not None:
        pm25_values = [None] * (len(plot_dates) - len(forecast_pm25)) + list(forecast_pm25)
        df['PM2.5_Forecast'] = pm25_values

    return df


# MAIN APP
def main():
    # Initialize session state for update tracking
    if 'update_checked' not in st.session_state:
        st.session_state.update_checked = False
        st.session_state.update_available = False
        st.session_state.new_version = None

    # Header
    st.markdown("""
        <div class="main-header">
            <h1>Air Quality Intelligence Dashboard</h1>
            <p>Real-time AQI Monitoring & 3-Day Forecast</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### System Status")

        # Load model
        with st.spinner("Loading AI model..."):
            model, version, error = load_champion_model()

        if error:
            st.cache_resource.clear()
            st.warning(f"Live model is not active yet. {error}")
        else:
            st.success("Live Model Active")
            st.info(f"**Model:** {MODEL_NAME}")
            st.info(f"**Version:** v{version}")
            st.info(f"**Registry:** DagsHub MLflow")

            # Auto-check for model updates on every app load
            if not st.session_state.update_checked and model:
                has_update, new_version, update_error = check_for_model_updates(version)
                st.session_state.update_checked = True
                st.session_state.update_available = has_update
                st.session_state.new_version = new_version

            # Display update notification if available
            if st.session_state.update_available and model:
                st.warning(f"New model available: v{st.session_state.new_version}")
                col1, col2 = st.columns(2)

                with col1:
                    if st.button("Update Now", width='stretch', type="primary"):
                        with st.spinner("Updating model..."):
                            st.cache_resource.clear()
                            st.session_state.update_available = False
                            st.session_state.update_checked = False
                            st.rerun()

                with col2:
                    if st.button("Skip", width='stretch'):
                        st.session_state.update_available = False
                        st.rerun()
            else:
                st.caption("Up to date" if model else "")

        st.markdown("---")

        # Manual refresh button
        if st.button("Refresh Model", width='stretch'):
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This dashboard provides:
        - **Real-time** air quality monitoring
        - **Interactive** Plotly visualizations
        - **Auto-update** model detection
        - **Health recommendations** based on AQI levels
        - **Historical trend** analysis
        - **Downloadable** reports
        """)

        st.markdown("---")
        st.caption("Developed by Shehraz Sarwar khan")
        st.caption("(a.k.a Data Scientist)")

    # Main content
    st.markdown("### Latest Air Quality Analysis")

    # Prediction button
    if st.button("Generate Forecast", width='stretch', type="primary"):

        with st.spinner("Fetching data from MongoDB..."):
            try:
                if MONGO_URI_ERROR:
                    raise RuntimeError(MONGO_URI_ERROR)

                # Connect to MongoDB
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
                client.admin.command("ping")
                db = client[DB_NAME]

                # Fetch historical data (7 days)
                history = list(db[RAW_COLLECTION].find().sort("datetime", -1).limit(168))  # 7 days hourly
                if not history:
                    st.error("No historical data found in database")
                    client.close()
                    st.stop()

                hist_df = pd.DataFrame(history).sort_values("datetime")
                hist_df['datetime'] = pd.to_datetime(hist_df['datetime'])
                present_time = hist_df['datetime'].iloc[-1]

                # Fetch latest features
                latest_feat_doc = list(db[FEATURE_COLLECTION].find().sort("datetime", -1).limit(1))
                if not latest_feat_doc:
                    st.error("No feature data found in database")
                    client.close()
                    st.stop()

                latest_feat_doc = latest_feat_doc[0]
                client.close()

            except Exception as e:
                error_message = str(e)
                if "bad auth" in error_message.lower() or "authentication failed" in error_message.lower():
                    st.error(
                        "MongoDB authentication failed. Check the Atlas username and password in .env. "
                        "URL-encode special password characters and add authSource=admin to the URI."
                    )
                else:
                    st.error(f"Database Error: {error_message}")
                st.stop()

        with st.spinner("Running prediction model..."):
            try:
                if model:
                    # Prepare features
                    feat_df = pd.DataFrame([latest_feat_doc]).drop(
                        columns=['_id', 'datetime', 'target_h24', 'target_h48', 'target_h72'],
                        errors='ignore'
                    )

                    # Make predictions
                    forecast_pm25 = np.maximum(model.predict(feat_df).flatten(), 0)
                else:
                    # Fallback estimate until the live champion model is registered and available.
                    recent_pm = hist_df.tail(72)['pm2_5'].mean()
                    forecast_pm25 = np.array([recent_pm * (1 + (i * 0.1)) for i in range(3)])

            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")
                # Fallback estimate while the live champion model is being restored.
                recent_pm = hist_df.tail(72)['pm2_5'].mean() if not hist_df.empty else 25
                forecast_pm25 = np.array([recent_pm * (1 + (i * 0.1)) for i in range(3)])

        # Process results
        plot_dates, aqi_values, types = [], [], []

        # Past 3 days + today (Observed) - total 4 days
        # Day -3, -2, -1, and 0 (today)
        for d in [3, 2, 1, 0]:
            t_date = (present_time - timedelta(days=d)).date()
            day_data = hist_df[hist_df['datetime'].dt.date == t_date]
            avg_pm = day_data['pm2_5'].mean() if not day_data.empty else 0
            aqi = calculate_aqi(avg_pm)
            # Convert to pandas Timestamp for consistency
            plot_dates.append(pd.Timestamp(datetime.combine(t_date, datetime.min.time())))
            aqi_values.append(aqi)
            types.append("Observed")

        # Next 3 days (Predicted) - starting from tomorrow
        # Tomorrow (+1 day), Day after (+2 days), Third day (+3 days)
        for i, days_ahead in enumerate([1, 2, 3]):
            f_dt = present_time + timedelta(days=days_ahead)
            aqi = calculate_aqi(forecast_pm25[i])
            # Convert to pandas Timestamp for consistency
            plot_dates.append(pd.Timestamp(f_dt))
            aqi_values.append(aqi)
            types.append("Predicted")

        # DISPLAY RESULTS
        st.success("Analysis Complete!")

        # Current status card (today's AQI - last observed point)
        cur_aqi = aqi_values[3]  # Today's observed (index 3: day 0)
        cat_name, cat_color, rec = get_aqi_info(cur_aqi)

        st.markdown(f"""
            <div class="alert-card" style="border-left-color: {cat_color};">
                <h3 style="color: {cat_color}; margin-top: 0;">
                    Current Air Quality: {cat_name} (AQI {cur_aqi})
                </h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">{rec}</p>
            </div>
        """, unsafe_allow_html=True)

        # Key metrics
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            cur_aqi_cat = get_aqi_info(cur_aqi)[0]
            st.metric(
                label="Today's AQI",
                value=f"{cur_aqi}",
                delta=cur_aqi_cat
            )

        with col2:
            forecast_tomorrow = aqi_values[4]  # Tomorrow (index 4)
            forecast_tomorrow_cat = get_aqi_info(forecast_tomorrow)[0]
            st.metric(
                label="Tomorrow",
                value=f"{forecast_tomorrow}",
                delta=forecast_tomorrow_cat
            )

        with col3:
            forecast_day2 = aqi_values[5]  # Day after tomorrow (index 5)
            forecast_day2_cat = get_aqi_info(forecast_day2)[0]
            st.metric(
                label="Day +2",
                value=f"{forecast_day2}",
                delta=forecast_day2_cat
            )

        with col4:
            forecast_day3 = aqi_values[6]  # Third day (index 6)
            forecast_day3_cat = get_aqi_info(forecast_day3)[0]
            st.metric(
                label="Day +3",
                value=f"{forecast_day3}",
                delta=forecast_day3_cat
            )

        # Interactive Plotly Visualization
        st.markdown("### Interactive AQI Trend Analysis")
        fig = create_aqi_chart_plotly(plot_dates, aqi_values, types)
        st.plotly_chart(fig, width='stretch')

        # Detailed information tabs
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Forecast Table", "Health Guidance", "Data Insights", "Historical Overview"])

        with tab1:
            st.markdown("#### Complete 7-Day AQI Report")

            # Create downloadable dataframe
            report_df = create_download_dataframe(plot_dates, aqi_values, types, forecast_pm25)

            # Display the dataframe
            st.dataframe(report_df, width='stretch', hide_index=True)

            # Download button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                # Convert to CSV
                csv_buffer = io.StringIO()
                report_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="Download 7-Day Report (CSV)",
                    data=csv_data,
                    file_name=f"AQI_7Day_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    width='stretch'
                )

        with tab2:
            st.markdown("#### Personalized Health Recommendations")

            for i in range(4, 7):  # Predicted days
                d_name = plot_dates[i].strftime('%A, %B %d')
                c_name, c_color, c_rec = get_aqi_info(aqi_values[i])

                st.markdown(f"""
                    <div class="alert-card" style="border-left-color: {c_color};">
                        <h4 style="color: {c_color}; margin-top: 0;">{d_name}</h4>
                        <p style="margin: 0;"><strong>Forecast:</strong> {c_name} (AQI {aqi_values[i]})</p>
                        <p style="margin: 0.5rem 0 0 0;">{c_rec}</p>
                    </div>
                """, unsafe_allow_html=True)

        with tab3:
            st.markdown("#### Statistical Summary")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Historical Data (Past 4 Days)**")
                hist_aqi = aqi_values[:4]
                st.write(f"• Average AQI: {np.mean(hist_aqi):.1f}")
                st.write(f"• Max AQI: {max(hist_aqi)}")
                st.write(f"• Min AQI: {min(hist_aqi)}")
                st.write(f"• Trend: {'Improving ↓' if hist_aqi[-1] < hist_aqi[0] else 'Worsening ↑'}")

            with col2:
                st.markdown("**Forecast (Next 3 Days)**")
                pred_aqi = aqi_values[4:]
                st.write(f"• Average AQI: {np.mean(pred_aqi):.1f}")
                st.write(f"• Max AQI: {max(pred_aqi)}")
                st.write(f"• Min AQI: {min(pred_aqi)}")
                st.write(f"• Overall Outlook: {get_aqi_info(int(np.mean(pred_aqi)))[0]}")

            st.markdown("---")
            st.info(f"Last Updated: {present_time.strftime('%Y-%m-%d %H:%M:%S')}")

            # Predicted PM2.5 values
            st.markdown("---")
            st.markdown("#### Predicted PM2.5 Concentrations")
            pred_df = pd.DataFrame({
                'Forecast Period': ['24 Hours', '48 Hours', '72 Hours'],
                'PM2.5 (μg/m³)': [f"{pm:.2f}" for pm in forecast_pm25],
                'AQI': [aqi_values[4], aqi_values[5], aqi_values[6]]
            })
            st.dataframe(pred_df, width='stretch', hide_index=True)

        with tab4:
            st.markdown("#### Environmental Parameters - Historical Overview")

            # Create interactive historical chart
            hist_chart = create_historical_overview_chart(hist_df)
            st.plotly_chart(hist_chart, width='stretch')

            # Summary statistics
            st.markdown("---")
            st.markdown("#### Summary Statistics (Past 7 Days)")

            col1, col2 = st.columns(2)

            # Handle both old and new column naming conventions
            temp_col = 'temp' if 'temp' in hist_df.columns else 'temperature'
            hum_col = 'rh' if 'rh' in hist_df.columns else 'humidity'

            with col1:
                st.metric("Avg PM2.5", f"{hist_df['pm2_5'].mean():.2f} μg/m³")
                if temp_col in hist_df.columns:
                    st.metric("Avg Temperature", f"{hist_df[temp_col].mean():.1f}°C")

            with col2:
                st.metric("Avg PM10", f"{hist_df['pm10'].mean():.2f} μg/m³")
                if hum_col in hist_df.columns:
                    st.metric("Avg Humidity", f"{hist_df[hum_col].mean():.1f}%")

    else:
        # Initial state - show instructions
        st.info("Click the button above to generate the latest air quality forecast")

        # Show example info
        st.markdown("### What You'll Get")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                **Real-time Analysis**
                - Current AQI status
                - PM2.5 measurements
                - 24-hour trends
            """)

        with col2:
            st.markdown("""
                **AI Predictions**
                - 24h forecast
                - 48h forecast
                - 72h forecast
            """)

        with col3:
            st.markdown("""
                **Health Guidance**
                - Category-based advice
                - Activity recommendations
                - Risk assessments
            """)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                **Interactive Visualizations**
                - Plotly-powered charts
                - Multi-parameter analysis
                - Historical trends
            """)

        with col2:
            st.markdown("""
                **Export Features**
                - Download 7-day reports
                - CSV format
                - Complete data export
            """)

if __name__ == "__main__":
    main()
