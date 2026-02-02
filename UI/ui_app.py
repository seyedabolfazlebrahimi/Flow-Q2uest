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
    "Choose spatial filter (HUC8 or County), then parameters, "
    "then explore stations and time series."
)

# --------------------------------------------------
# Network helpers (Retry)
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


# --------------------------------------------------
# Cached loaders
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_huc8_list(api_base: str):
    payload = get_json_with_retry(f"{api_base}/huc8-list")
    return payload.get("huc8_values", [])


@st.cache_data(ttl=3600)
def load_county_list(api_base: str):
    payload = get_json_with_retry(f"{api_base}/county-list")
    return payload.get("counties", [])


@st.cache_data(ttl=3600)
def load_parameters(api_base: str, huc8s, counties):
    params = build_query_params(huc8s=huc8s, counties=counties)
    payload = get_json_with_retry(f"{api_base}/parameters", params=params)
    return payload.get("parameter_column"), payload.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(api_base, parameters, huc8s, counties, max_points=20000, min_samples=1):
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
    df["MonitoringLocationIdentifierCor"] = df["MonitoringLocationIdentifierCor"].astype(str)

    if "n_samples" in df.columns:
        df["n_samples"] = pd.to_numeric(df["n_samples"], errors="coerce")

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
    params = build_query_params(
        parameters=[parameter],
        huc8s=huc8s,
        counties=counties,
    )
    payload = get_json_with_retry(f"{api_base}/mean-per-station", params=params)
    return pd.DataFrame(payload)


# --------------------------------------------------
# Time series utilities (UNCHANGED)
# --------------------------------------------------
def parse_timeseries(df_raw):
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Date", "CuratedResultMeasureValue"])

    df = df_raw.copy()

    date_candidates = ["ActivityStartDate", "ActivityStartDateTime", "Date"]
    value_candidates = ["CuratedResultMeasureValue", "ResultMeasureValue", "Value"]

    date_col = next((c for c in date_candidates if c in df.columns), None)
    value_col = next((c for c in value_candidates if c in df.columns), None)

    if date_col is None or value_col is None:
        return pd.DataFrame(columns=["Date", "CuratedResultMeasureValue"])

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["CuratedResultMeasureValue"] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["Date", "CuratedResultMeasureValue"]).sort_values("Date")
    return df[["Date", "CuratedResultMeasureValue"]]


def compute_mean_from_timeseries(df_ts):
    if df_ts.empty:
        return None
    return float(df_ts["CuratedResultMeasureValue"].mean())


def render_timeseries_chart(df_ts, height=350):
    chart_df = df_ts.rename(columns={"CuratedResultMeasureValue": "Concentration"})
    base = alt.Chart(chart_df).encode(
        x=alt.X("Date:T", title="Year"),
        y=alt.Y("Concentration:Q", title="Concentration"),
        tooltip=["Date:T", "Concentration:Q"],
    )
    st.altair_chart(
        (base.mark_line() + base.mark_point(size=40)).properties(height=height),
        use_container_width=True,
    )


# ==================================================
# UI
# ==================================================

# ----------------------------
# 0) Spatial filter
# ----------------------------
st.header("0) Spatial Filter")

mode = st.radio("Filter by:", ["HUC8", "County"], horizontal=True)

selected_huc8s = []
selected_counties = []

if mode == "HUC8":
    selected_huc8s = st.multiselect("Select HUC8 basin(s)", load_huc8_list(API_BASE))
else:
    selected_counties = st.multiselect("Select County / Counties", load_county_list(API_BASE))

st.divider()

# ----------------------------
# 1) Select Parameter
# ----------------------------
param_col, params = load_parameters(API_BASE, selected_huc8s, selected_counties)

selected_params = st.multiselect("Choose parameter(s)", params, default=params[:1])
plot_param = st.selectbox("Plot parameter", selected_params)

min_samples_ui = st.number_input("Minimum samples per station", min_value=1, value=1)

st.divider()

# ----------------------------
# 2) Stations (Table)
# ----------------------------
st.header("2) Stations")

if st.button("Load stations"):
    df_st = load_stations_df(
        API_BASE,
        selected_params,
        selected_huc8s,
        selected_counties,
        min_samples=min_samples_ui,
    )
    st.dataframe(df_st, use_container_width=True)

# ----------------------------
# 3) Station Picker
# ----------------------------
st.header("3) Station Picker")

df_picker = load_stations_df(
    API_BASE,
    selected_params,
    selected_huc8s,
    selected_counties,
    min_samples=min_samples_ui,
)

station_id_input = st.selectbox(
    "Select station",
    df_picker["MonitoringLocationIdentifierCor"].tolist() if not df_picker.empty else [],
)

st.divider()

# ----------------------------
# 4) Station data -> CSV (+ chart + mean)
# ----------------------------
st.header("4) Station Data → CSV (with Time Series Plot)")

if st.button("Fetch station data"):
    df_raw = load_station_data(
        API_BASE,
        station_id_input,
        selected_params,
        selected_huc8s,
        selected_counties,
    )

    df_plot_raw = df_raw[df_raw[param_col] == plot_param] if param_col else df_raw
    df_ts = parse_timeseries(df_plot_raw)

    mean_val = compute_mean_from_timeseries(df_ts)

    st.markdown(f"**Mean:** {mean_val:.4g}" if mean_val else "**Mean:** N/A")
    render_timeseries_chart(df_ts)
    st.dataframe(df_raw, use_container_width=True)

# ----------------------------
# 7) Mean Concentration Map
# ----------------------------
st.header("7) Mean Concentration Map")

if st.button("Show mean maps"):
    df_coords = load_stations_df(
        API_BASE,
        selected_params,
        selected_huc8s,
        selected_counties,
        min_samples=min_samples_ui,
    )

    df_mean = load_mean_per_station(
        API_BASE,
        plot_param,
        selected_huc8s,
        selected_counties,
    )

    df_join = df_mean.merge(df_coords, on="MonitoringLocationIdentifierCor")

    m = folium.Map(location=[df_join["lat"].mean(), df_join["lon"].mean()], zoom_start=7)
    for _, r in df_join.iterrows():
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=5,
            tooltip=f"{r['MonitoringLocationIdentifierCor']} | mean={r['mean_value']:.3g}",
            fill=True,
        ).add_to(m)

    st_folium(m, width=1000, height=600)