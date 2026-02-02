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

# Use env var if set (Render), otherwise fallback to local
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").strip()

st.set_page_config(page_title="FLOW Q2UEST (Demo)", layout="wide")
st.title("FLOW Q2UEST (Demo)")
st.markdown(
    "Interactive UI connected to the Water Quality API. "
    "Choose a spatial filter (HUC8 or County), then choose one or more parameters, "
    "then explore stations and time series."
)

# ----------------------------
# Network helpers (Retry)
# ----------------------------
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


# ----------------------------
# Helpers
# ----------------------------
def build_query_params(
    station_id=None,
    parameters=None,
    huc8s=None,
    counties=None,  # ✅ NEW
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
            if str(h).strip():
                params.append(("huc8", str(h).strip()))

    if counties:
        for c in counties:
            if str(c).strip():
                params.append(("county", str(c).strip()))

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


# ----------------------------
# Cached loaders
# ----------------------------
@st.cache_data(ttl=3600)
def load_huc8_list(api_base: str):
    payload = get_json_with_retry(f"{api_base}/huc8-list")
    return payload.get("huc8_column"), payload.get("huc8_values", [])


@st.cache_data(ttl=3600)
def load_county_list(api_base: str):
    payload = get_json_with_retry(f"{api_base}/county-list")
    return payload.get("county_column"), payload.get("county_values", [])


@st.cache_data(ttl=3600)
def load_parameters(api_base: str, huc8s, counties):
    params = build_query_params(huc8s=huc8s, counties=counties)
    payload = get_json_with_retry(f"{api_base}/parameters", params=params)
    return payload.get("parameter_column"), payload.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(api_base, parameters, huc8s, counties, max_points, min_samples):
    params = build_query_params(
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
        max_points=max_points,
        min_samples=min_samples,
    )
    payload = get_json_with_retry(f"{api_base}/stations", params=params)
    df = pd.DataFrame(payload)

    if df.empty:
        return df

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df["MonitoringLocationIdentifierCor"] = df["MonitoringLocationIdentifierCor"].astype(str).str.strip()

    if "n_samples" in df.columns:
        df["n_samples"] = pd.to_numeric(df.get("n_samples"), errors="coerce")

    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=900)
def load_station_data(api_base, station_id, parameters, huc8s, counties):
    params = build_query_params(
        station_id=station_id,
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
    )
    payload = get_json_with_retry(f"{api_base}/station-data", params=params)
    return pd.DataFrame(_unpack_station_payload(payload))


@st.cache_data(ttl=900)
def load_mean_per_station(api_base, parameter, huc8s, counties):
    params = build_query_params(parameters=[parameter], huc8s=huc8s, counties=counties)
    payload = get_json_with_retry(f"{api_base}/mean-per-station", params=params)
    return pd.DataFrame(payload)

# ======================================================
# 0) Spatial Filter (HUC8 / County)
# ======================================================
st.header("0) Spatial Filter")

mode = st.radio(
    "Filter stations by:",
    options=["HUC8", "County"],
    horizontal=True,
)

selected_huc8s = []
selected_counties = []

if mode == "HUC8":
    huc8_col, huc8_values = load_huc8_list(API_BASE)
    selected_huc8s = st.multiselect(
        "Choose one or more HUC8 basins (leave empty for ALL)",
        options=huc8_values,
        default=[],
    )
else:
    county_col, county_values = load_county_list(API_BASE)
    selected_counties = st.multiselect(
        "Choose one or more counties (leave empty for ALL)",
        options=county_values,
        default=[],
    )

selected_huc8s_t = tuple(selected_huc8s)
selected_counties_t = tuple(selected_counties)
st.divider()