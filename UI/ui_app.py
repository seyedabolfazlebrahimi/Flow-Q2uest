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
    "Choose a spatial filter (HUC8 or County), then parameters, "
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
    counties=None,   # ✅ NEW
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
    return pd.DataFrame(payload)


@st.cache_data(ttl=900)
def load_station_data(api_base, station_id, parameters, huc8s, counties):
    params = build_query_params(
        station_id=station_id,
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
    )
    payload = get_json_with_retry(f"{api_base}/station-data", params=params)
    if isinstance(payload, dict) and "data" in payload:
        return pd.DataFrame(payload["data"])
    return pd.DataFrame(payload)


# ======================================================
# 0) Spatial filter selector (NEW)
# ======================================================
st.header("0) Spatial Filter")

filter_mode = st.radio(
    "Search stations by:",
    options=["HUC8", "County Name"],
    horizontal=True,
)

selected_huc8s = []
selected_counties = []

if filter_mode == "HUC8":
    huc8_col, huc8_values = load_huc8_list(API_BASE)
    selected_huc8s = st.multiselect(
        "Select HUC8 basin(s)",
        options=huc8_values,
        default=[],
    )
else:
    county_col, county_values = load_county_list(API_BASE)
    selected_counties = st.multiselect(
        "Select County name(s)",
        options=county_values,
        default=[],
    )

st.divider()

# ----------------------------
# 1) Select Parameter
# ----------------------------
param_col, params = load_parameters(
    API_BASE,
    huc8s=selected_huc8s,
    counties=selected_counties,
)

selected_params = st.multiselect(
    "Select parameter(s)",
    options=params,
    default=params[:1] if params else [],
)

plot_param = st.selectbox("Plot parameter", selected_params)

min_samples_ui = st.number_input("Minimum samples per station", min_value=1, value=1)

st.divider()

# ----------------------------
# 2) Stations Table
# ----------------------------
if st.button("Load stations"):
    df_st = load_stations_df(
        API_BASE,
        parameters=selected_params,
        huc8s=selected_huc8s,
        counties=selected_counties,
        max_points=20000,
        min_samples=min_samples_ui,
    )
    st.dataframe(df_st, use_container_width=True)

# ----------------------------
# 3) Station Picker
# ----------------------------
df_picker = load_stations_df(
    API_BASE,
    parameters=selected_params,
    huc8s=selected_huc8s,
    counties=selected_counties,
    max_points=20000,
    min_samples=min_samples_ui,
)

if not df_picker.empty:
    df_picker["label"] = (
        df_picker["MonitoringLocationIdentifierCor"].astype(str)
        + " | n=" + df_picker["n_samples"].astype(str)
    )

    selected_label = st.selectbox("Select station", df_picker["label"])
    station_id = selected_label.split("|")[0].strip()

    if st.button("Load station data"):
        df_raw = load_station_data(
            API_BASE,
            station_id,
            selected_params,
            selected_huc8s,
            selected_counties,
        )
        st.dataframe(df_raw, use_container_width=True)