import requests
import pandas as pd
import streamlit as st # type: ignore
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path
from typing import Optional

# =========================================================
# Configuration & Constants
# =========================================================

# Page Configuration
st.set_page_config(
    page_title="Traffic AI Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths and API endpoints
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "US_Accidents_50k.csv"

VISION_API_URL = "http://127.0.0.1:8000/api/v1/vision/predict"
NLP_API_URL = "http://127.0.0.1:8000/nlp/predict"
DURATION_API_URL = "http://127.0.0.1:8000/api/v1/duration/predict" # Ensure this is correctly defined


# =========================================================
# Custom CSS - Professional Dark Theme
# =========================================================
def apply_custom_css():
    """Applies custom CSS for a professional dark theme."""
    st.markdown(
        """
        <style>
            :root {
                --bg: #0B1020;
                --surface: #151C31;
                --surface-2: #1D2742;
                --border: #2D3B5E;
                --text: #F3F7FF;
                --muted: #A8B3CF;
                --accent: #47B5FF;
                --accent-2: #7C4DFF;
                --success: #22C55E;
                --warning: #F59E0B;
                --danger: #EF4444;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(71, 181, 255, 0.12), transparent 25%),
                    radial-gradient(circle at top right, rgba(124, 77, 255, 0.10), transparent 20%),
                    linear-gradient(180deg, #0B1020 0%, #0E1426 100%);
                color: var(--text);
            }

            header[data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"] {
                background: var(--bg) !important;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #11182B 0%, #0F1627 100%);
                border-right: 1px solid rgba(255, 255, 255, 0.06);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 1rem;
                max-width: 1500px;
            }

            .hero-card {
                background: linear-gradient(135deg, rgba(71, 181, 255, 0.16), rgba(124, 77, 255, 0.12));
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 22px 24px;
                margin-bottom: 16px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.18);
            }
            .hero-title { font-size: 2rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.35rem; }
            .hero-subtitle { color: var(--muted); font-size: 0.98rem; line-height: 1.55; }

            .section-card {
                background: rgba(21, 28, 49, 0.92);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 18px;
                padding: 16px;
                margin-bottom: 14px;
                box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
            }

            .mini-card {
                background: rgba(29, 39, 66, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                padding: 16px;
                min-height: 100px;
            }
            .metric-title { color: var(--muted); font-size: 0.85rem; margin-bottom: 8px; }
            .metric-value { color: #FFFFFF; font-size: 1.8rem; font-weight: 800; line-height: 1; }
            .metric-help { color: #8EA2C9; font-size: 0.78rem; margin-top: 6px; }

            .status-chip {
                display: inline-block; padding: 6px 12px; border-radius: 999px;
                font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(255, 255, 255, 0.08);
                margin-top: 6px;
            }
            .status-ok { background: rgba(34, 197, 94, 0.15); color: #86EFAC; }
            .status-warn { background: rgba(245, 158, 11, 0.15); color: #FCD34D; }
            .status-danger { background: rgba(239, 68, 68, 0.15); color: #FCA5A5; }

            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            .stTextInput input, .stNumberInput input, .stTextArea textarea {
                background-color: #11182B !important; color: #F3F7FF !important;
                border: 1px solid #2D3B5E !important; border-radius: 12px !important;
            }

            .stTabs [data-baseweb="tab-list"] { gap: 8px; }
            .stTabs [data-baseweb="tab"] {
                background: rgba(29, 39, 66, 0.9); border: 1px solid rgba(255, 255, 255, 0.05);
                color: #DCE7FF; border-radius: 12px; padding: 10px 16px;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(71, 181, 255, 0.20), rgba(124, 77, 255, 0.18));
                border: 1px solid rgba(71, 181, 255, 0.45); color: #FFFFFF;
            }

            .api-box {
                background: rgba(11, 16, 32, 0.75); border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
            }

            .small-muted { color: var(--muted); font-size: 0.85rem; }
            .footer-note { color: #91A2C7; font-size: 0.82rem; margin-top: 6px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Apply CSS on load
apply_custom_css()

# =========================================================
# Utility Functions
# =========================================================

def section_title(title: str, subtitle: Optional[str] = None) -> None:
    """Renders a card with a section title and optional subtitle."""
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)
    st.markdown('</div>', unsafe_allow_html=True)

def metric_card(title: str, value: str, help_text: str = "") -> None:
    """Renders a small card displaying a key metric."""
    st.markdown(
        f"""
        <div class="mini-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def status_chip(label: str, mode: str = "ok") -> None:
    """Renders a status indicator chip (e.g., OK, Warning, Danger)."""
    css_class = {
        "ok": "status-ok",
        "warn": "status-warn",
        "danger": "status-danger",
    }.get(mode, "status-ok")
    st.markdown(
        f'<span class="status-chip {css_class}">{label}</span>',
        unsafe_allow_html=True,
    )

@st.cache_data(show_spinner="Loading accident dataset...")
def load_accident_data() -> pd.DataFrame:
    """Loads the local US Accidents dataset using Pandas. Handles potential errors."""
    try:
        return pd.read_csv(DATASET_PATH, low_memory=False)
    except FileNotFoundError:
        st.error(f"Dataset not found at: {DATASET_PATH}")
        st.stop()
    except Exception as error:
        st.error(f"Could not load dataset: {error}")
        st.stop()

def safe_datetime_column(df: pd.DataFrame, column_name: str) -> pd.Series:
    """Safely converts a column to datetime objects, coercing errors."""
    return pd.to_datetime(df[column_name], errors="coerce")

def detect_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Finds the first column from a list of candidates that exists in the DataFrame."""
    for col in candidates:
        if col in df.columns:
            return col
    return None

def render_hero() -> None:
    """Renders the main hero section with title and subtitle."""
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Traffic Incident Intelligence Platform</div>
            <div class="hero-subtitle">
                An end-to-end traffic incident analytics platform that combines computer vision,
                 NLP-based severity prediction, and incident duration forecasting through
                  an interactive Streamlit dashboard powered by FastAPI services.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# EDA Page Rendering
# =========================================================
def render_eda_page() -> None:
    """Renders the Exploratory Data Analysis page."""
    # Add extra spacing at the top to push content down
    st.write("")
    st.write("")
    st.write("")

    df = load_accident_data()
    if df.empty:
        st.warning("The dataset was loaded, but it is empty.")
        st.stop()

    # --- Column Detection ---
    severity_col = detect_existing_column(df, ["Severity"])
    weather_col = detect_existing_column(df, ["Weather_Condition"])
    visibility_col = detect_existing_column(df, ["Visibility(mi)", "Visibility_mi"])
    wind_col = detect_existing_column(df, ["Wind_Speed(mph)", "Wind_Speed_mph"])
    temp_col = detect_existing_column(df, ["Temperature(F)", "Temperature_F"])
    humidity_col = detect_existing_column(df, ["Humidity(%)", "Humidity_pct"])
    state_col = detect_existing_column(df, ["State"])
    city_col = detect_existing_column(df, ["City"])
    start_time_col = detect_existing_column(df, ["Start_Time"])
    lat_col = detect_existing_column(df, ["Start_Lat", "Latitude"])
    lng_col = detect_existing_column(df, ["Start_Lng", "Longitude"])

    # --- Datetime Feature Engineering ---
    if start_time_col:
        dt = safe_datetime_column(df, start_time_col)
        df["_hour"] = dt.dt.hour
        df["_month"] = dt.dt.month
        df["_weekday"] = dt.dt.day_name()
    else:
        df["_hour"], df["_month"], df["_weekday"] = None, None, None

    # --- KPI Calculations ---
    total_records = len(df)
    missing_rate = df.isna().mean().mean() * 100
    avg_severity = df[severity_col].mean() if severity_col else None
    top_weather = df[weather_col].astype("string").value_counts().index[0] if weather_col and df[weather_col].notna().any() else "N/A"
    top_state = df[state_col].astype("string").value_counts().index[0] if state_col and df[state_col].notna().any() else "N/A"

    # --- Tabbed Layout ---
    tab_home, tab_data, tab_overview, tab_severity, tab_weather, tab_time, tab_geo = st.tabs(
        ["Home", "Data", "Overview", "Severity", "Weather", "Time", "Geo"]
    )

    # --- Home Tab ---
    with tab_home:
        render_hero()
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("EDA Dashboard")
        st.caption("Interactive analytics dashboard built on the US Accidents dataset, providing key insights into traffic incidents, environmental conditions, and accident severity.")
        st.markdown('</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1: metric_card("Total Incidents", f"{total_records:,}", "Records loaded from the project dataset")
        with c2: metric_card("Average Severity", f"{avg_severity:.2f}" if avg_severity is not None else "N/A", "Mean accident severity level")
        with c3: metric_card("Top Weather", str(top_weather), "Most common weather condition")
        with c4: metric_card("Missing Rate", f"{missing_rate:.1f}%", f"State with the highest number of incidents: {top_state}")

        st.markdown(f'<div class="footer-note">Source: Local US Accidents Dataset (500K Records)</div>', unsafe_allow_html=True)

    # --- Data Tab ---
    with tab_data:
        left, right = st.columns([2.2, 1.2])
        with left:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Dataset Preview")
            st.dataframe(df.head(50), height=280, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with right:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Schema")
            schema_df = pd.DataFrame({
                "column": df.columns,
                "dtype": [str(dtype) for dtype in df.dtypes],
                "missing_%": [round(df[col].isna().mean() * 100, 2) for col in df.columns],
            })
            st.dataframe(schema_df, height=280, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Overview Tab ---
    with tab_overview:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Missing Values by Top Columns")
            missing_df = df.isna().mean().sort_values(ascending=False).head(12).mul(100).round(2).reset_index()
            missing_df.columns = ["column", "missing_pct"]
            fig = px.bar(missing_df, x="missing_pct", y="column", orientation="h", template="plotly_dark", height=340, color="missing_pct", color_continuous_scale="Blues")
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Top States by Incident Count")
            if state_col:
                top_states = df[state_col].astype("string").value_counts().head(12).reset_index()
                top_states.columns = ["State", "Count"]
                fig = px.bar(top_states, x="State", y="Count", template="plotly_dark", height=340, color="Count", color_continuous_scale="Purples")
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
            else: st.info("State column was not found in the dataset.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Severity Tab ---
    with tab_severity:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Severity Distribution")
            if severity_col:
                fig = px.histogram(df, x=severity_col, nbins=10, template="plotly_dark", height=340, color_discrete_sequence=["#47B5FF"])
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
            else: st.info("Severity column was not found in the dataset.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Severity vs Visibility")
            if severity_col and visibility_col:
                plot_df = df[[severity_col, visibility_col]].dropna().sample(n=min(15000, len(df)), random_state=42)
                if not plot_df.empty:
                    fig = px.box(plot_df, x=severity_col, y=visibility_col, template="plotly_dark", height=340, color=severity_col)
                    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
                    st.plotly_chart(fig, width="stretch")
                else: st.info("Not enough complete rows for severity-visibility analysis.")
            else: st.info("Required columns for severity-visibility analysis are missing.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Weather Tab ---
    with tab_weather:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Top Weather Conditions")
            if weather_col:
                weather_df = df[weather_col].astype("string").value_counts().head(12).reset_index()
                weather_df.columns = ["Weather", "Count"]
                fig = px.bar(weather_df, x="Count", y="Weather", orientation="h", template="plotly_dark", height=340, color="Count", color_continuous_scale="Tealgrn")
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
            else: st.info("Weather_Condition column was not found.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Weather Feature Correlation")
            corr_candidates = [col for col in [temp_col, humidity_col, visibility_col, wind_col] if col is not None]
            if len(corr_candidates) >= 2:
                corr_df = df[corr_candidates].dropna().sample(n=min(10000, len(df)), random_state=42)
                if not corr_df.empty:
                    corr_matrix = corr_df.corr(numeric_only=True)
                    fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.index, colorscale="Blues", text=corr_matrix.round(2).values, texttemplate="%{text}"))
                    fig.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=20, b=10))
                    st.plotly_chart(fig, width="stretch")
                else: st.info("Not enough complete rows to compute correlation.")
            else: st.info("Not enough weather-related numeric columns were found.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Time Tab ---
    with tab_time:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Accident Hours")
            if "_hour" in df.columns and df["_hour"].notna().any():
                hour_df = df["_hour"].dropna().astype(int).value_counts().sort_index().reset_index()
                hour_df.columns = ["Hour", "Count"]
                fig = px.line(hour_df, x="Hour", y="Count", markers=True, template="plotly_dark", height=340)
                fig.update_traces(line_color="#47B5FF")
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
                st.plotly_chart(fig, width="stretch")
            else: st.info("Start_Time column is missing or could not be parsed.")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown("#### Incidents by Weekday")
            if "_weekday" in df.columns and df["_weekday"].notna().any():
                weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                weekday_df = df["_weekday"].dropna().astype("string").value_counts().reindex(weekday_order).fillna(0).reset_index()
                weekday_df.columns = ["Weekday", "Count"]
                fig = px.bar(weekday_df, x="Weekday", y="Count", template="plotly_dark", height=340, color="Count", color_continuous_scale="Plasma")
                fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), coloraxis_showscale=False)
                st.plotly_chart(fig, width="stretch")
            else: st.info("Weekday information is unavailable.")
            st.markdown('</div>', unsafe_allow_html=True)

    # --- Geo Tab ---
    with tab_geo:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### Geographic Sample")
        if lat_col and lng_col:
            geo_df = df[[lat_col, lng_col]].dropna().sample(n=min(3000, len(df)), random_state=42)
            if not geo_df.empty:
                st.map(geo_df.rename(columns={lat_col: "lat", lng_col: "lon"}), height=380)
            else: st.info("No complete latitude/longitude rows were found.")
        else: st.info("Latitude/longitude columns were not found for geographic visualization.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Vision & Severity Page Rendering
# =========================================================
def render_vision_severity_page() -> None:
    """Renders the Vision and Severity prediction page."""
    st.write("") # Spacing
    st.write("")
    st.subheader("Vision & Severity Prediction")
    st.caption("Upload a road image first. If an accident is detected by the vision model, you can submit structured incident context and a manual text description to the NLP API.")

    tab_process, tab_api = st.tabs(["Vision & Prediction", "API Endpoints"])

    with tab_process:
        left, right = st.columns([1, 1], gap="medium")

        # --- Vision Prediction Section ---
        with left:
            st.markdown("##### Step 1 — Vision Prediction")
            uploaded_image = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png"], key="vision_image_uploader")
            if uploaded_image: st.image(uploaded_image, caption="Uploaded image", width=250)

            run_vision = st.button("Run Vision API", width="stretch", key="run_vision_button")

            if run_vision:
                if uploaded_image is None:
                    st.warning("Please upload an image first.")
                else:
                    try:
                        files = {"file": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
                        with st.spinner("Calling Vision API..."):
                            response = requests.post(VISION_API_URL, files=files, timeout=60)
                            if response.status_code == 200:
                                st.session_state["vision_result"] = response.json()
                                st.success("Vision prediction completed.")
                            else:
                                st.error(f"Vision API error: {response.status_code}")
                                try: st.json(response.json())
                                except Exception: st.text(response.text)
                    except Exception as error:
                        st.error(f"Could not call Vision API: {error}")

            vision_result = st.session_state.get("vision_result")
            if vision_result:
                accident_detected = vision_result.get("accident_detected", False)
                status_chip("Accident detected", "danger") if accident_detected else status_chip("No accident detected", "ok")
                with st.expander("Show Vision API Response", expanded=False): st.json(vision_result)
            else:
                status_chip("Waiting for image prediction", "warn")

        # --- NLP Severity Section ---
        with right:
            st.markdown("##### Step 2 — Severity Prediction")
            vision_result = st.session_state.get("vision_result")

            if not vision_result:
                status_chip("Waiting for vision response", "warn")
                st.info("Run the Vision API first.")
            else:
                accident_detected = vision_result.get("accident_detected", False)
                if not accident_detected:
                    status_chip("No accident detected", "ok")
                    st.success("The image was classified as non-accident. NLP inference is skipped.")
                else:
                    status_chip("Accident detected", "danger")
                    st.caption("Provide incident context for NLP severity prediction.")

                    with st.form("nlp_form"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            hour = st.number_input("Hour", min_value=0, max_value=23, value=14)
                            month = st.number_input("Month", min_value=1, max_value=12, value=10)
                            weekday = st.number_input("Weekday", min_value=0, max_value=6, value=4)
                        with col2:
                            temperature = st.number_input("Temperature_F", value=72.0)
                            humidity = st.number_input("Humidity_pct", value=58.0)
                        with col3:
                            visibility = st.number_input("Visibility_mi", value=10.0)
                            wind_speed = st.number_input("Wind_Speed_mph", value=7.0)

                        description = st.text_area("Description", height=80, placeholder="Example: Multi-vehicle collision on highway with blocked left lane and slow traffic.")
                        submit_nlp = st.form_submit_button("Run NLP API", use_container_width=True)

                    if submit_nlp:
                        payload = {
                            "Hour": int(hour), "Month": int(month), "Weekday": int(weekday),
                            "Temperature_F": float(temperature), "Humidity_pct": float(humidity),
                            "Visibility_mi": float(visibility), "Wind_Speed_mph": float(wind_speed),
                            "Description": description,
                        }
                        try:
                            with st.spinner("Calling NLP API..."):
                                response = requests.post(NLP_API_URL, json=payload, timeout=60)
                                if response.status_code == 200:
                                    result = response.json()
                                    st.session_state["nlp_result"] = result
                                    st.success("NLP prediction completed.")
                                    with st.expander("Show NLP API Response", expanded=True): st.json(result)
                                else:
                                    st.error(f"NLP API error: {response.status_code}")
                                    try: st.json(response.json())
                                    except Exception: st.text(response.text)
                        except Exception as error:
                            st.error(f"Could not call NLP API: {error}")

    # --- API Endpoints Section ---
    with tab_api:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### API Endpoints")
        api_col1, api_col2 = st.columns(2)
        with api_col1:
            st.markdown(f'<div class="api-box"><strong>Vision API</strong><br><code>{VISION_API_URL}</code></div>', unsafe_allow_html=True)
        with api_col2:
            st.markdown(f'<div class="api-box"><strong>NLP API</strong><br><code>{NLP_API_URL}</code></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Incident Duration Page Rendering
# =========================================================
def render_duration_page() -> None:
    """Renders the Incident Duration prediction page."""
    st.write("") # Spacing
    st.write("")
    st.subheader("Incident Duration Prediction")
    st.caption("Submit structured accident attributes to estimate incident duration in minutes. Datetime format must be: YYYY-MM-DD HH:MM:SS")
    st.markdown('</div>', unsafe_allow_html=True) # Closing the implicit section-card from st.subheader

    tab_process, tab_api_endpoint = st.tabs(["Duration Prediction", "API Endpoint"])

    with tab_process:
        # Using columns and expander to reduce vertical space
        c1, c2, c3 = st.columns(3)
        with st.form("duration_form", clear_on_submit=True):
            with c1:
                start_time = st.text_input("Start_Time", value="2023-10-27 14:30:00")
                severity = st.number_input("Severity", min_value=1, max_value=4, value=2)
                distance_mi = st.number_input("Distance(mi)", min_value=0.0, value=0.5)
                temperature = st.number_input("Temperature(F)", value=72.0)
                humidity = st.number_input("Humidity(%)", value=58.0)

            with c2:
                visibility = st.number_input("Visibility(mi)", value=10.0)
                wind_speed = st.number_input("Wind_Speed(mph)", value=7.0)
                pressure = st.number_input("Pressure(in)", value=29.92)
                precipitation = st.number_input("Precipitation(in)", value=0.0)
                weather_condition = st.text_input("Weather_Condition", value="Fair")

            with c3:
                state = st.text_input("State", value="CA")
                sunrise_sunset = st.selectbox("Sunrise_Sunset", ["Day", "Night"])

                with st.expander("Boolean Incident Attributes", expanded=False):
                    # Boolean attributes now correctly placed within the expander
                    amenity = st.selectbox("Amenity", [False, True], index=0)
                    bump = st.selectbox("Bump", [False, True], index=0)
                    crossing = st.selectbox("Crossing", [False, True], index=0)
                    junction = st.selectbox("Junction", [False, True], index=0)
                    traffic_signal = st.selectbox("Traffic_Signal", [False, True], index=0)
                    give_way = st.selectbox("Give_Way", [False, True], index=0)
                    no_exit = st.selectbox("No_Exit", [False, True], index=0)
                    railway = st.selectbox("Railway", [False, True], index=0)
                    roundabout = st.selectbox("Roundabout", [False, True], index=0)
                    station = st.selectbox("Station", [False, True], index=0)
                    stop_sign = st.selectbox("Stop", [False, True], index=0) # Renamed from Stop to Stop_Sign for clarity if needed by API, check API docs
                    traffic_calming = st.selectbox("Traffic_Calming", [False, True], index=0)

            submitted = st.form_submit_button("Run Duration API", use_container_width=True)

        if submitted:
            # Constructing the payload with correct field names
            payload = {
                "Start_Time": start_time, "Severity": int(severity),
                "Distance(mi)": float(distance_mi), "Temperature(F)": float(temperature),
                "Humidity(%)": float(humidity), "Visibility(mi)": float(visibility),
                "Wind_Speed(mph)": float(wind_speed), "Pressure(in)": float(pressure),
                "Precipitation(in)": float(precipitation), "Weather_Condition": weather_condition,
                "State": state, "Sunrise_Sunset": sunrise_sunset,
                # Boolean values
                "Amenity": bool(amenity), "Bump": bool(bump), "Crossing": bool(crossing),
                "Give_Way": bool(give_way), "Junction": bool(junction), "No_Exit": bool(no_exit),
                "Railway": bool(railway), "Roundabout": bool(roundabout), "Station": bool(station),
                "Stop": bool(stop_sign), "Traffic_Calming": bool(traffic_calming),
                "Traffic_Signal": bool(traffic_signal),
            }

            try:
                with st.spinner("Calling Duration API..."):
                    if not DURATION_API_URL:
                        st.error("DURATION_API_URL is not defined. Please set it.")
                    else:
                        response = requests.post(DURATION_API_URL, json=payload, timeout=60)
                        if response.status_code == 200:
                            result = response.json()
                            st.markdown('<div class="section-card">', unsafe_allow_html=True)
                            st.markdown("#### Prediction Result")
                            col1_res, col2_res = st.columns(2)
                            with col1_res:
                                metric_card("Predicted Duration", f"{result.get('predicted_duration_minutes', 'N/A')}", "Minutes")
                            with col2_res:
                                metric_card("Readable Format", str(result.get("predicted_duration_readable", "N/A")), "Human-friendly output")
                            st.json(result)
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Duration API error: {response.status_code}")
                            try: st.json(response.json())
                            except Exception: st.text(response.text)
            except requests.exceptions.RequestException as error:
                st.error(f"Could not connect to Duration API: {error}")
            except Exception as error:
                st.error(f"An unexpected error occurred: {error}")

    with tab_api_endpoint:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("#### API Endpoint")
        st.markdown(f'<div class="api-box"><strong>Duration API:</strong> <code>{DURATION_API_URL}</code></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# About Page Rendering
# =========================================================
def render_about_page() -> None:
    """Renders the About Project page with organized information in tabs."""
    st.write("") # Spacing
    st.write("")
    st.subheader("About Project")
    st.caption("System overview, architecture, models, and technology stack.")
    st.markdown('</div>', unsafe_allow_html=True) # Close the implicit section-card

    tab_goal, tab_modules, tab_tech, tab_api, tab_notes = st.tabs(
        ["Project Goal", "Core Modules", "Technology Stack", "API Routes", "Notes"]
    )

    with tab_goal:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Project Goal")
        st.write("""This project is an end-to-end traffic incident intelligence system designed to:
            - detect traffic accidents from road images,
            - estimate severity from textual context,
            - predict incident duration from structured accident features,
            - and provide an operational dashboard for analysis and demonstration.""")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_modules:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Core Modules")
        st.write("""
            **1. Vision / CNN**
            - Image-based accident vs non-accident classification

            **2. NLP Severity Prediction**
            - Baseline approach: **Bag-of-Words + CountVectorizer**
            - Text description + structured weather/time features

            **3. Incident Duration Prediction**
            - Structured tabular prediction for duration in minutes
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_tech:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Technology Stack")
        st.write("""
            - **Frontend:** Streamlit
            - **Backend:** FastAPI
            - **Model Serving:** Uvicorn
            - **EDA & Visualization:** Pandas, Plotly
            - **Vision Model:** CNN / TensorFlow-Keras
            - **NLP Baseline:** CountVectorizer-based Bag-of-Words
            - **ML Utilities:** Scikit-learn, XGBoost
            """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_api:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### API Routes")
        st.code(
            f"""
POST {VISION_API_URL.replace("http://localhost:8000", "")}/predict
POST {NLP_API_URL.replace("http://localhost:8000", "")}/predict
POST {DURATION_API_URL.replace("http://localhost:8000", "")}/predict
            """.strip(),
            language="text",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with tab_notes:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### Notes")
        st.write("""
            - The frontend is intentionally compact to reduce unnecessary scrolling.
            - EDA uses the local dataset file from the project folder.
            - Vision and NLP are linked sequentially: NLP is only triggered when an accident is detected.
            - Current API integration focuses on endpoint health and valid JSON responses.
            """)
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# Sidebar Navigation
# =========================================================
with st.sidebar:
    st.markdown("## 🚦 Traffic AI")
    st.markdown('<div class="small-muted">Traffic Incident Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["EDA", "Vision & Severity", "Incident Duration", "About Project"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Environment")
    st.markdown("""
        - FastAPI
        - Streamlit
        - US Accidents (500K)
        """)

# =========================================================
# Main Page Routing
# =========================================================
if page == "EDA":
    render_eda_page()
elif page == "Vision & Severity":
    render_vision_severity_page()
elif page == "Incident Duration":
    render_duration_page()
else: # About Project
    render_about_page()
