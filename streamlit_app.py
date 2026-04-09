# streamlit_app.py
import streamlit as st
import requests
import pandas as pd
import altair as alt
from datetime import datetime

# -------------------------------
# API config
# -------------------------------
API_URL = "http://127.0.0.1:8000"
DATA_PATH = "data_processed/sales_ca1_clean.csv"

# -------------------------------
# Load unique_ids for dropdown
# -------------------------------
df_all = pd.read_csv(DATA_PATH)
all_unique_ids = sorted(df_all["unique_id"].astype(str).unique())

# Initialize session state for series selection
if "selected_series" not in st.session_state:
    st.session_state.selected_series = []

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.title("Forecast Settings")

# Horizon selection
horizon = st.sidebar.selectbox("Select Horizon", options=[7, 30])

# Top-K or specific series
top_k_toggle = st.sidebar.checkbox("Top-K series?")
if top_k_toggle:
    top_k = st.sidebar.slider("Select Top-K series", min_value=1, max_value=20, value=5)
    unique_ids = None
else:
    top_k = None
    # Select All / Deselect All buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All"):
            st.session_state.selected_series = all_unique_ids
    with col2:
        if st.button("Deselect All"):
            st.session_state.selected_series = []

    unique_ids = st.sidebar.multiselect(
        "Select series (unique_id)",
        options=all_unique_ids,
        default=st.session_state.selected_series
    )
    # Update session state
    st.session_state.selected_series = unique_ids

# Run forecast button
run_forecast = st.sidebar.button("Run Forecast")

# -------------------------------
# Main Panel
# -------------------------------
st.title("M5 Sales Forecast Dashboard")

if run_forecast:
    # Build API parameters
    params = {"horizon": horizon}
    if unique_ids:
        params["unique_ids"] = unique_ids
    if top_k:
        params["top_k"] = top_k

    # Call API
    try:
        response = requests.get(f"{API_URL}/forecast", params=params)
        response.raise_for_status()
        res_json = response.json()
        data = pd.DataFrame(res_json["forecast"])
        model_version = res_json["model_version"]
    except Exception as e:
        st.error(f"Error fetching forecast: {e}")
        st.stop()

    # Display metadata
    st.markdown(f"**Model Version:** {model_version}")
    st.markdown(f"**Horizon:** {horizon}d")
    st.markdown(f"**Run Timestamp:** {datetime.now().isoformat()}")

    # Display table of forecasts
    st.subheader("Forecast Table")
    st.dataframe(data)

    # Plot each series
    st.subheader("Historical vs Forecast")

    # Rename columns for plotting
    data_plot = data.rename(columns={
        "ds": "day",
        "y": "real sales",
        "y_pred_rounded": "predicted sales",
        "naive_pred": "naive sales"
    })

    for series in data_plot["unique_id"].unique():
        df_s = data_plot[data_plot["unique_id"] == series].sort_values("day")
        
        # Melt dataframe for Altair-friendly format
        df_melt = df_s.melt(
            id_vars=["day"], 
            value_vars=["real sales", "predicted sales", "naive sales"],
            var_name="type", 
            value_name="sales"
        )
        
        # Define consistent colors
        color_scale = alt.Scale(
            domain=["real sales", "predicted sales", "naive sales"],
            range=["blue", "red", "green"]
        )
        
        # Line chart
        line_chart = (
            alt.Chart(df_melt)
            .mark_line()
            .encode(
                x=alt.X(
                    "day:T",
                    title="Day",
                    axis=alt.Axis(format="%d/%m", labelAngle=-45, tickCount=len(df_s))
                ),
                y=alt.Y(
                    "sales:Q",
                    title="Sales",
                    axis=alt.Axis(format="d"),
                    scale=alt.Scale(nice=False)
                ),
                color=alt.Color("type:N", scale=color_scale, title="Legend")
            )
        )
        
        # Point chart
        point_chart = (
            alt.Chart(df_melt)
            .mark_point(filled=True, size=100)
            .encode(
                x="day:T",
                y="sales:Q",
                color=alt.Color("type:N", scale=color_scale),
                tooltip=["day:T", "type:N", "sales:Q"]
            )
        )
        
        combined_chart = (line_chart + point_chart).interactive()
        st.altair_chart(combined_chart, use_container_width=True)