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
    "Explore stations, time series, spatial patterns, and mean concentration maps."
)

# --------------------------------------------------
# Network helpers
# --------------------------------------------------
def get_json_with_retry(url, params=None, tries=5, timeout=180):
    last_exc = None
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_exc = e
            time.sleep(1.2 * (i + 1))
    raise last_exc


# --------------------------------------------------
# Query params
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


# --------------------------------------------------
# Safe unpack
# --------------------------------------------------
def _unpack(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    return []


# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_huc8_list(api):
    return get_json_with_retry(f"{api}/huc8-list").get("huc8_values", [])


@st.cache_data(ttl=3600)
def load_county_list(api):
    return get_json_with_retry(f"{api}/county-list").get("counties", [])


@st.cache_data(ttl=3600)
def load_parameters(api, huc8s, counties):
    params = build_query_params(huc8s=huc8s, counties=counties)
    payload = get_json_with_retry(f"{api}/parameters", params=params)
    return payload.get("parameter_column"), payload.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(api, parameters, huc8s, counties, min_samples):
    params = build_query_params(
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
        min_samples=min_samples,
    )
    payload = get_json_with_retry(f"{api}/stations", params=params)
    df = pd.DataFrame(payload)
    if df.empty:
        return df
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=900)
def load_station_data(api, station_id, parameters, huc8s, counties):
    params = build_query_params(
        station_id=station_id,
        parameters=parameters,
        huc8s=huc8s,
        counties=counties,
    )
    payload = get_json_with_retry(f"{api}/station-data", params=params)
    return pd.DataFrame(_unpack(payload))


# --------------------------------------------------
# ✅ FIXED: Mean per station (robust)
# --------------------------------------------------
@st.cache_data(ttl=1800)
def load_mean_per_station(api, parameter, huc8s, counties):
    params = build_query_params(
        parameters=[parameter],
        huc8s=huc8s,
        counties=counties,
    )

    endpoints = [
        "/mean-per-station",
        "/station-mean",
        "/mean_by_station",
    ]

    last_exc = None
    for ep in endpoints:
        try:
            payload = get_json_with_retry(f"{api}{ep}", params=params)
            return pd.DataFrame(payload)
        except Exception as e:
            last_exc = e

    raise last_exc


# --------------------------------------------------
# Time series utils
# --------------------------------------------------
def parse_timeseries(df):
    if df.empty:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")
    df["Value"] = pd.to_numeric(df["CuratedResultMeasureValue"], errors="coerce")
    df = df.dropna(subset=["Date", "Value"]).sort_values("Date")
    return df[["Date", "Value"]]


def render_ts(df):
    chart = alt.Chart(df).mark_line(point=True).encode(
        x="Date:T",
        y="Value:Q",
        tooltip=["Date:T", "Value:Q"],
    )
    st.altair_chart(chart, use_container_width=True)


# ==================================================
# UI
# ==================================================
st.header("0) Spatial Filter")

mode = st.radio("Filter by:", ["HUC8", "County"], horizontal=True)

selected_huc8s, selected_counties = [], []

if mode == "HUC8":
    selected_huc8s = st.multiselect(
        "Select HUC8 basin(s)",
        load_huc8_list(API_BASE),
    )
else:
    counties = load_county_list(API_BASE)
    selected_counties = st.multiselect("Select counties", counties)

st.divider()

# --------------------------------------------------
st.header("1) Parameter")

param_col, parameters = load_parameters(
    API_BASE,
    selected_huc8s,
    selected_counties,
)

selected_param = st.selectbox("Select parameter", parameters)

min_samples = st.number_input(
    "Minimum samples per station", min_value=1, value=1
)

st.divider()

# --------------------------------------------------
st.header("2) Stations")

df_st = load_stations_df(
    API_BASE,
    [selected_param],
    selected_huc8s,
    selected_counties,
    min_samples,
)

st.dataframe(df_st, use_container_width=True)

st.divider()

# --------------------------------------------------
st.header("3) Station Time Series")

station_id = st.selectbox(
    "Select station",
    df_st["MonitoringLocationIdentifierCor"].astype(str).tolist(),
)

df_raw = load_station_data(
    API_BASE,
    station_id,
    [selected_param],
    selected_huc8s,
    selected_counties,
)

df_ts = parse_timeseries(df_raw)

render_ts(df_ts)
st.markdown(f"**Mean:** {df_ts['Value'].mean():.3g}")

st.divider()

# --------------------------------------------------
st.header("4) Station Map")

m = folium.Map(
    location=[df_st["lat"].mean(), df_st["lon"].mean()],
    zoom_start=7,
)

for _, r in df_st.iterrows():
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=4,
        tooltip=r["MonitoringLocationIdentifierCor"],
        fill=True,
    ).add_to(m)

st_folium(m, width=1000, height=550)

st.divider()

# --------------------------------------------------
st.header("5) Mean Concentration Map")

df_mean = load_mean_per_station(
    API_BASE,
    selected_param,
    selected_huc8s,
    selected_counties,
)

df_mean = df_mean.merge(
    df_st,
    on="MonitoringLocationIdentifierCor",
    how="inner",
)

vmin, vmax = df_mean["mean_value"].min(), df_mean["mean_value"].max()

m2 = folium.Map(
    location=[df_mean["lat"].mean(), df_mean["lon"].mean()],
    zoom_start=7,
)

for _, r in df_mean.iterrows():
    color = "#ff0000" if r["mean_value"] > (vmin + vmax) / 2 else "#0000ff"
    folium.CircleMarker(
        location=[r["lat"], r["lon"]],
        radius=6,
        color=color,
        fill=True,
        tooltip=f"{r['MonitoringLocationIdentifierCor']} | mean={r['mean_value']:.3g}",
    ).add_to(m2)

st_folium(m2, width=1100, height=600)