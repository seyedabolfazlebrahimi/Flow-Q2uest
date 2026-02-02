# filename: ui_app.py
import os
import json
import time

import pandas as pd
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium
import altair as alt

# --------------------------------------------------
# Config
# --------------------------------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").strip()

st.set_page_config(page_title="FLOW Q2UEST (Demo)", layout="wide")
st.title("FLOW Q2UEST (Demo)")
st.markdown(
    "Interactive UI connected to the Water Quality API. "
    "Choose a spatial filter (**HUC8** or **County**), then choose one or more parameters, "
    "then explore stations and time series."
)

# --------------------------------------------------
# Network helpers
# --------------------------------------------------
def get_json_with_retry(url, params=None, tries=6, timeout=180):
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (i + 1))
    raise last_exc


# --------------------------------------------------
# Query helpers
# --------------------------------------------------
def build_query_params(
    station_id=None,
    parameters=None,
    huc8s=None,
    counties=None,
    start_year=None,
    end_year=None,
    max_points=None,
    min_samples=None,
):
    params = []

    if station_id:
        params.append(("station_id", station_id))

    if parameters:
        for p in parameters:
            if str(p).strip():
                params.append(("parameter", str(p).strip()))

    if huc8s:
        for h in huc8s:
            params.append(("huc8", str(h)))

    if counties:
        for c in counties:
            params.append(("county", str(c)))

    if start_year is not None:
        params.append(("start_year", int(start_year)))
    if end_year is not None:
        params.append(("end_year", int(end_year)))

    if max_points is not None:
        params.append(("max_points", int(max_points)))
    if min_samples is not None:
        params.append(("min_samples", int(min_samples)))

    return params


def _unpack_station_payload(payload):
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return []


# --------------------------------------------------
# API loaders
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_huc8_list(api_base):
    p = get_json_with_retry(f"{api_base}/huc8-list")
    return p.get("huc8_column"), p.get("huc8_values", [])


@st.cache_data(ttl=3600)
def load_county_list(api_base):
    p = get_json_with_retry(f"{api_base}/county-list")
    return p.get("counties", [])


@st.cache_data(ttl=3600)
def load_parameters(api_base, huc8s, counties):
    params = build_query_params(huc8s=list(huc8s), counties=list(counties))
    p = get_json_with_retry(f"{api_base}/parameters", params=params)
    return p.get("parameter_column"), p.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(api_base, parameters, huc8s, counties, max_points=20000, min_samples=1):
    params = build_query_params(
        parameters=list(parameters),
        huc8s=list(huc8s),
        counties=list(counties),
        max_points=max_points,
        min_samples=min_samples,
    )
    df = pd.DataFrame(get_json_with_retry(f"{api_base}/stations", params=params))
    if df.empty:
        return df

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["MonitoringLocationIdentifierCor"] = df["MonitoringLocationIdentifierCor"].astype(str)

    if "n_samples" in df.columns:
        df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")

    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=900)
def load_station_data(api_base, station_id, huc8s, counties):
    # 🚫 NO parameter passed here (critical fix)
    params = build_query_params(
        station_id=station_id,
        huc8s=list(huc8s),
        counties=list(counties),
    )
    payload = get_json_with_retry(f"{api_base}/station-data", params=params, timeout=360)
    return pd.DataFrame(_unpack_station_payload(payload))


@st.cache_data(ttl=900)
def load_mean_per_station(api_base, parameter, huc8s, counties):
    params = build_query_params(parameters=[parameter], huc8s=list(huc8s), counties=list(counties))
    return pd.DataFrame(get_json_with_retry(f"{api_base}/mean-per-station", params=params))


# --------------------------------------------------
# Timeseries helpers
# --------------------------------------------------
def parse_timeseries(df):
    if df.empty:
        return df

    date_col = next((c for c in ["ActivityStartDate", "Date", "SampleDate"] if c in df.columns), None)
    val_col = next((c for c in ["CuratedResultMeasureValue", "ResultMeasureValue", "Value"] if c in df.columns), None)

    if not date_col or not val_col:
        return pd.DataFrame()

    out = df[[date_col, val_col]].copy()
    out.columns = ["Date", "Value"]
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    return out.dropna().sort_values("Date")


def plot_ts(df):
    if df.empty:
        st.info("No data to plot.")
        return
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Date:T",
            y="Value:Q",
            tooltip=["Date:T", "Value:Q"],
        )
    )
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------
# UI
# --------------------------------------------------
st.header("0) Spatial Filter")
mode = st.radio("Filter by:", ["HUC8", "County"], horizontal=True)

selected_huc8s, selected_counties = [], []

if mode == "HUC8":
    _, huc8_vals = load_huc8_list(API_BASE)
    selected_huc8s = st.multiselect("Select HUC8 (optional)", huc8_vals)
else:
    counties = load_county_list(API_BASE)
    selected_counties = st.multiselect("Select County (optional)", counties)

selected_huc8s_t = tuple(selected_huc8s)
selected_counties_t = tuple(selected_counties)

st.divider()

st.header("1) Select Parameter")
param_col, params = load_parameters(API_BASE, selected_huc8s_t, selected_counties_t)
selected_params = st.multiselect("Parameters", params, default=params[:1])
plot_param = st.selectbox("Plot parameter", selected_params)

min_samples = st.number_input("Minimum samples per station", min_value=1, value=1)

st.divider()

st.header("3) Station Picker")
df_picker = load_stations_df(
    API_BASE, selected_params, selected_huc8s_t, selected_counties_t, min_samples=min_samples
)

df_picker["label"] = df_picker["MonitoringLocationIdentifierCor"]
station_id = st.selectbox("Select station", df_picker["label"])

st.divider()

st.header("4) Station Data → Time Series")

if st.button("Fetch station data"):
    df_raw = load_station_data(API_BASE, station_id, selected_huc8s_t, selected_counties_t)

    df_plot = df_raw[df_raw[param_col] == plot_param] if param_col in df_raw.columns else df_raw
    df_ts = parse_timeseries(df_plot)

    plot_ts(df_ts)
    st.dataframe(df_raw)

st.divider()

st.header("6) Station Map + Time Series")

if st.button("Show map"):
    df_map = df_picker.head(5000)
    m = folium.Map(location=[df_map.lat.mean(), df_map.lon.mean()], zoom_start=7)

    for _, r in df_map.iterrows():
        folium.CircleMarker(
            location=[r.lat, r.lon],
            radius=4,
            tooltip=r.MonitoringLocationIdentifierCor,
        ).add_to(m)

    st_folium(m, width=900, height=500)