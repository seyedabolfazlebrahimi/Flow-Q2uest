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
    "Filter by **HUC8 or County**, then choose parameters, explore stations, "
    "time series, and spatial patterns."
)

# --------------------------------------------------
# Network helpers (robust retry)
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
# Helpers
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
            params.append(("parameter", str(p)))

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
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return []


# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_huc8_list(api_base):
    payload = get_json_with_retry(f"{api_base}/huc8-list")
    return payload.get("huc8_values", [])


@st.cache_data(ttl=3600)
def load_county_list(api_base):
    payload = get_json_with_retry(f"{api_base}/county-list")
    return payload.get("counties", [])


@st.cache_data(ttl=3600)
def load_parameters(api_base, huc8s, counties):
    params = build_query_params(huc8s=huc8s, counties=counties)
    payload = get_json_with_retry(f"{api_base}/parameters", params=params)
    return payload.get("parameter_column"), payload.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(api_base, parameters, huc8s, counties, min_samples):
    params = build_query_params(
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
        min_samples=min_samples,
    )
    payload = get_json_with_retry(f"{api_base}/stations", params=params)
    df = pd.DataFrame(payload)
    if df.empty:
        return df
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
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
    rows = _unpack_station_payload(payload)
    return pd.DataFrame(rows)


# --------------------------------------------------
# Time series utilities
# --------------------------------------------------
def parse_timeseries(df):
    if df.empty:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
    df["Value"] = pd.to_numeric(df["CuratedResultMeasureValue"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"]).sort_values("Date")
    return df[["Date", "Value"]]


def render_timeseries_chart(df):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y("Value:Q", title="Concentration"),
        tooltip=["Date:T", "Value:Q"],
    )
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------
# 0) Spatial Filter
# --------------------------------------------------
st.header("0) Spatial Filter")

mode = st.radio(
    "Filter by:",
    ["HUC8", "County"],
    horizontal=True,
)

selected_huc8s = []
selected_counties = []

if mode == "HUC8":
    huc8_values = load_huc8_list(API_BASE)
    selected_huc8s = st.multiselect(
        "Select HUC8 basin(s)",
        options=huc8_values,
        default=[],
    )
else:
    all_counties = load_county_list(API_BASE)
    county_search = st.text_input(
        "Filter counties (optional)",
        placeholder="e.g. ala, palm, brow",
    )

    if county_search.strip():
        filtered = [
            c for c in all_counties
            if county_search.lower() in c.lower()
        ]
    else:
        filtered = all_counties

    selected_counties = st.multiselect(
        "Select County / Counties",
        options=filtered,
        default=[],
    )

st.divider()

# --------------------------------------------------
# 1) Select Parameter
# --------------------------------------------------
param_col, parameters = load_parameters(
    API_BASE,
    selected_huc8s,
    selected_counties,
)

if not parameters:
    st.warning("No parameters found for the selected spatial filter.")
    st.stop()

selected_param = st.selectbox("Select parameter", parameters)

min_samples = st.number_input(
    "Minimum samples per station",
    min_value=1,
    value=1,
    step=1,
)

st.divider()

# --------------------------------------------------
# 2) Stations table
# --------------------------------------------------
st.header("2) Stations")

if st.button("Load stations"):
    df_st = load_stations_df(
        API_BASE,
        [selected_param],
        selected_huc8s,
        selected_counties,
        min_samples,
    )
    if df_st.empty:
        st.warning("No stations found.")
    else:
        st.dataframe(df_st, use_container_width=True)

st.divider()

# --------------------------------------------------
# 3) Station picker + time series
# --------------------------------------------------
st.header("3) Station Time Series")

df_picker = load_stations_df(
    API_BASE,
    [selected_param],
    selected_huc8s,
    selected_counties,
    min_samples,
)

if df_picker.empty:
    st.warning("No stations available.")
    st.stop()

station_id = st.selectbox(
    "Select station",
    df_picker["MonitoringLocationIdentifierCor"].astype(str).tolist(),
)

df_raw = load_station_data(
    API_BASE,
    station_id,
    [selected_param],
    selected_huc8s,
    selected_counties,
)

if df_raw.empty:
    st.warning("No data for this station.")
else:
    df_ts = parse_timeseries(df_raw)
    render_timeseries_chart(df_ts)
    st.dataframe(df_raw, use_container_width=True)

st.divider()

# --------------------------------------------------
# 4) Map
# --------------------------------------------------
st.header("4) Station Map")

df_map = df_picker.head(5000)

if not df_map.empty:
    m = folium.Map(
        location=[df_map["lat"].mean(), df_map["lon"].mean()],
        zoom_start=7,
    )

    for _, r in df_map.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=4,
            tooltip=r["MonitoringLocationIdentifierCor"],
            fill=True,
        ).add_to(m)

    st_folium(m, width=1000, height=600)
