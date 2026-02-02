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

# ----------------------------
# 1) Select Parameter - MULTI
# ----------------------------
st.header("1) Select Parameter")

param_col, params = load_parameters(API_BASE, selected_huc8s_t, selected_counties_t)

if not params:
    st.error("No parameters found for the selected spatial filter.")
    st.stop()

selected_params = st.multiselect(
    "Choose one or more parameters",
    options=params,
    default=[params[0]] if params else [],
)

if not selected_params:
    st.warning("Please select at least one parameter to continue.")
    st.stop()

selected_params_t = tuple(selected_params)
plot_param = st.selectbox("Plot parameter (for charts)", options=list(selected_params))

st.caption("Station filter")
min_samples_ui = st.number_input(
    "Minimum samples per station",
    min_value=1,
    value=1,
    step=1,
)

st.divider()

# ----------------------------
# 2) Stations (Table)
# ----------------------------
st.header("2) Stations (Table)")

with st.expander("Show stations list (map-ready)"):
    if st.button("Load stations"):
        df_st = load_stations_df(
            API_BASE,
            selected_params_t,
            selected_huc8s_t,
            selected_counties_t,
            max_points=20000,
            min_samples=min_samples_ui,
        )
        if df_st.empty:
            st.warning("No stations returned for current filters.")
        else:
            if "n_samples" in df_st.columns:
                st.caption("`n_samples` = number of rows per station AFTER filters.")
            st.dataframe(df_st, use_container_width=True)
            st.download_button(
                "Download stations as CSV",
                df_st.to_csv(index=False).encode("utf-8"),
                file_name="stations.csv",
                mime="text/csv",
            )

# ----------------------------
# 3) Station Picker
# ----------------------------
st.header("3) Station Picker")

with st.spinner("Loading stations for picker..."):
    df_picker = load_stations_df(
        API_BASE,
        selected_params_t,
        selected_huc8s_t,
        selected_counties_t,
        max_points=20000,
        min_samples=min_samples_ui,
    )

if df_picker.empty:
    st.warning("No stations available for current filters.")
    st.stop()

df_picker = df_picker.copy()

if "n_samples" in df_picker.columns:
    df_picker["label"] = (
        df_picker["MonitoringLocationIdentifierCor"].astype(str)
        + " | n=" + df_picker["n_samples"].fillna(0).astype(int).astype(str)
        + " | lat=" + df_picker["lat"].round(3).astype(str)
        + ", lon=" + df_picker["lon"].round(3).astype(str)
    )
else:
    df_picker["label"] = (
        df_picker["MonitoringLocationIdentifierCor"].astype(str)
        + " | lat=" + df_picker["lat"].round(3).astype(str)
        + ", lon=" + df_picker["lon"].round(3).astype(str)
    )

selected_label = st.selectbox(
    "Select a station",
    options=df_picker["label"].tolist(),
    index=0,
)

station_id_input = selected_label.split("|")[0].strip()
st.caption(f"Selected station_id: **{station_id_input}**")

st.divider()

# ----------------------------
# 4) Station data → CSV (+ plot)
# ----------------------------
st.header("4) Station Data → CSV (with Time Series Plot)")

col1, col2 = st.columns(2)

with col1:
    if st.button("Fetch station data"):
        df_raw = load_station_data(
            API_BASE,
            station_id_input,
            selected_params_t,
            selected_huc8s_t,
            selected_counties_t,
        )

        if df_raw.empty:
            st.warning("No data returned for this station.")
        else:
            st.success(f"Rows: {len(df_raw)}")

            if param_col and param_col in df_raw.columns:
                df_plot = df_raw[
                    df_raw[param_col].astype(str).str.strip()
                    == str(plot_param).strip()
                ]
            else:
                df_plot = df_raw.copy()

            if not df_plot.empty:
                df_plot["Date"] = pd.to_datetime(
                    df_plot.get("ActivityStartDate", df_plot.get("Date")),
                    errors="coerce",
                )
                df_plot["Value"] = pd.to_numeric(
                    df_plot.get("CuratedResultMeasureValue", df_plot.get("Value")),
                    errors="coerce",
                )
                df_plot = df_plot.dropna(subset=["Date", "Value"]).sort_values("Date")

                if not df_plot.empty:
                    mean_val = df_plot["Value"].mean()
                    st.markdown(f"**Mean:** {mean_val:.4g}")

                    chart = alt.Chart(df_plot).mark_line(point=True).encode(
                        x="Date:T",
                        y="Value:Q",
                        tooltip=["Date:T", "Value:Q"],
                    )
                    st.altair_chart(chart, use_container_width=True)

            st.dataframe(df_raw, use_container_width=True)
            st.download_button(
                "Download station CSV",
                df_raw.to_csv(index=False).encode("utf-8"),
                file_name=f"station_{station_id_input}.csv",
                mime="text/csv",
            )

with col2:
    st.info("Station selected from picker above.")

# ----------------------------
# 5) Download by year range
# ----------------------------
st.header("5) Download by Year Range")

c1, c2, c3 = st.columns([1, 1, 2])
start_year = c1.number_input("Start year", value=2000)
end_year = c2.number_input("End year", value=2010)

if c3.button("Download CSV"):
    params = build_query_params(
        parameters=list(selected_params_t),
        huc8s=list(selected_huc8s_t),
        counties=list(selected_counties_t),
        start_year=start_year,
        end_year=end_year,
    )
    r = requests.get(f"{API_BASE}/download-range", params=params, timeout=360)
    st.download_button(
        "Download subset.csv",
        r.content,
        file_name="subset.csv",
        mime="text/csv",
    )

# ----------------------------
# 6) Map + Time Series
# ----------------------------
st.header("6) Station Map + Time Series")

df_map = load_stations_df(
    API_BASE,
    selected_params_t,
    selected_huc8s_t,
    selected_counties_t,
    max_points=5000,
    min_samples=min_samples_ui,
)

if not df_map.empty:
    center_lat = float(df_map["lat"].mean())
    center_lon = float(df_map["lon"].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

    for _, row in df_map.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            tooltip=row["MonitoringLocationIdentifierCor"],
            fill=True,
        ).add_to(m)

    st_folium(m, width=900, height=550)
else:
    st.info("No stations to display on map.")

# ----------------------------
# 7) Mean Concentration Map
# ----------------------------
st.header("7) Mean Concentration Map")

for p in selected_params_t:
    st.subheader(f"Mean Map: {p}")

    df_mean = load_mean_per_station(
        API_BASE,
        p,
        selected_huc8s_t,
        selected_counties_t,
    )

    if df_mean.empty:
        st.warning(f"No mean data for {p}")
        continue

    df_join = df_mean.merge(
        df_map,
        on="MonitoringLocationIdentifierCor",
        how="inner",
    ).dropna(subset=["lat", "lon", "mean_value"])

    if df_join.empty:
        st.warning("No stations with mean + coordinates.")
        continue

    vmin = df_join["mean_value"].min()
    vmax = df_join["mean_value"].max()

    m = folium.Map(
        location=[df_join["lat"].mean(), df_join["lon"].mean()],
        zoom_start=7,
    )

    for _, row in df_join.iterrows():
        color = "#ff0000"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color=color,
            fill=True,
            tooltip=row["MonitoringLocationIdentifierCor"],
        ).add_to(m)

    st_folium(m, width=1100, height=600)