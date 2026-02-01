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

# ----------------------------
# Config: column name for County
# ----------------------------
COUNTY_COL = "CountyName"   # your dataset column name

st.set_page_config(page_title="FLOW Q2UEST (Demo)", layout="wide")
st.title("FLOW Q2UEST (Demo)")
st.markdown(
    "Interactive UI connected to the Water Quality API. "
    "Choose **HUC8 or County** (optional), then choose one or more parameters, "
    "then explore stations and time series."
)

# ----------------------------
# Network helpers (Retry)
# ----------------------------
def get_json_with_retry(url, params=None, tries=6, timeout=180):
    """
    Render Free can intermittently return 502 (cold start / restart / proxy).
    This wrapper retries with backoff.
    """
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
    counties=None,            # ✅ NEW
    start_year=None,
    end_year=None,
    max_points=None,
    min_samples=None,  # ✅ NEW
):
    """
    Build request params allowing repeated keys:
    params = [("parameter", p1), ("parameter", p2), ("huc8", h1), ("county", c1), ...]
    """
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

    # ✅ NEW: counties
    if counties:
        for c in counties:
            if str(c).strip():
                params.append(("county", str(c).strip()))

    if start_year is not None:
        params.append(("start_year", int(start_year)))
    if end_year is not None:
        params.append(("end_year", int(end_year)))

    # cap stations returned by API (prevents heavy payload + makes UI stable)
    if max_points is not None:
        params.append(("max_points", int(max_points)))

    # ✅ minimum samples per station (server-side filter)
    if min_samples is not None:
        params.append(("min_samples", int(min_samples)))

    return params


def _unpack_station_payload(payload):
    """
    API /station-data might return:
      A) list[dict]   (old)
      B) dict with {"data": list[dict], ...}  (new)
    This function returns list[dict] safely.
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        return []

    if isinstance(payload, str):
        try:
            obj = json.loads(payload)
            return _unpack_station_payload(obj)
        except Exception:
            return []

    return []


@st.cache_data(ttl=3600)
def load_huc8_list(api_base: str):
    payload = get_json_with_retry(f"{api_base}/huc8-list", params=None, tries=6, timeout=180)
    return payload.get("huc8_column"), payload.get("huc8_values", [])


# ✅ NEW: county list loader
@st.cache_data(ttl=3600)
def load_county_list(api_base: str):
    """
    Expects API endpoint:
      GET /county-list -> {"county_column": "...", "counties": [...], "source": "..."}
    """
    payload = get_json_with_retry(f"{api_base}/county-list", params=None, tries=6, timeout=240)
    return payload.get("county_column", COUNTY_COL), payload.get("counties", [])


@st.cache_data(ttl=3600)
def load_parameters(api_base: str, huc8s: tuple[str, ...], counties: tuple[str, ...]):
    params = build_query_params(huc8s=list(huc8s), counties=list(counties))
    payload = get_json_with_retry(f"{api_base}/parameters", params=params, tries=6, timeout=240)
    return payload.get("parameter_column"), payload.get("parameters", [])


@st.cache_data(ttl=3600)
def load_stations_df(
    api_base: str,
    parameters: tuple[str, ...],
    huc8s: tuple[str, ...],
    counties: tuple[str, ...],          # ✅ NEW
    max_points: int = 20000,
    min_samples: int = 1,  # ✅ NEW
) -> pd.DataFrame:
    params = build_query_params(
        parameters=list(parameters),
        huc8s=list(huc8s),
        counties=list(counties),
        max_points=max_points,
        min_samples=min_samples,
    )
    payload = get_json_with_retry(f"{api_base}/stations", params=params, tries=6, timeout=240)

    df = pd.DataFrame(payload)
    if df.empty:
        return df

    # If API returns {"error": "..."} as dict -> DataFrame will have weird columns; handle quickly
    if "error" in df.columns:
        return pd.DataFrame()

    df["lat"] = pd.to_numeric(df.get("lat"), errors="coerce")
    df["lon"] = pd.to_numeric(df.get("lon"), errors="coerce")
    df["MonitoringLocationIdentifierCor"] = df["MonitoringLocationIdentifierCor"].astype(str).str.strip()

    # n_samples optional (new API); keep if exists
    if "n_samples" in df.columns:
        df["n_samples"] = pd.to_numeric(df.get("n_samples"), errors="coerce")

    return df.dropna(subset=["lat", "lon"])


@st.cache_data(ttl=900)
def load_station_data(
    api_base: str,
    station_id: str,
    parameters: tuple[str, ...],
    huc8s: tuple[str, ...],
    counties: tuple[str, ...],           # ✅ NEW
) -> pd.DataFrame:
    station_id = str(station_id).strip()
    params = build_query_params(
        station_id=station_id,
        parameters=list(parameters),
        huc8s=list(huc8s),
        counties=list(counties),
    )
    payload = get_json_with_retry(f"{api_base}/station-data", params=params, tries=6, timeout=360)

    rows = _unpack_station_payload(payload)
    return pd.DataFrame(rows)


@st.cache_data(ttl=900)
def load_mean_per_station(api_base: str, parameter: str, huc8s: tuple[str, ...], counties: tuple[str, ...]):
    params = build_query_params(parameters=[parameter], huc8s=list(huc8s), counties=list(counties))
    payload = get_json_with_retry(f"{api_base}/mean-per-station", params=params, tries=6, timeout=360)
    return pd.DataFrame(payload)


def parse_timeseries(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust timeseries parser (never KeyError):
    - Detects date + value columns safely
    - Returns standardized columns: Date, CuratedResultMeasureValue
    """
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Date", "CuratedResultMeasureValue"])

    df = df_raw.copy()

    date_candidates = [
        "ActivityStartDate",
        "ActivityStartDateTime",
        "Date",
        "SampleDate",
        "sample_date",
    ]
    date_col = next((c for c in date_candidates if c in df.columns), None)

    value_candidates = [
        "CuratedResultMeasureValue",
        "ResultMeasureValue",
        "Value",
        "result_value",
    ]
    value_col = next((c for c in value_candidates if c in df.columns), None)

    if date_col is None or value_col is None:
        return pd.DataFrame(columns=["Date", "CuratedResultMeasureValue"])

    df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["CuratedResultMeasureValue"] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.dropna(subset=["Date", "CuratedResultMeasureValue"]).sort_values("Date")
    return df[["Date", "CuratedResultMeasureValue"]]


def compute_mean_from_timeseries(df_ts: pd.DataFrame):
    if df_ts is None or df_ts.empty or "CuratedResultMeasureValue" not in df_ts.columns:
        return None
    vals = pd.to_numeric(df_ts["CuratedResultMeasureValue"], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def render_timeseries_chart(df_ts: pd.DataFrame, height: int = 350) -> None:
    if df_ts is None or df_ts.empty:
        st.info("No timeseries data to plot.")
        return

    chart_df = df_ts.rename(columns={"CuratedResultMeasureValue": "Concentration"}).copy()

    base = alt.Chart(chart_df).encode(
        x=alt.X("Date:T", axis=alt.Axis(title="Year", format="%Y", tickCount="year")),
        y=alt.Y("Concentration:Q", title="Concentration"),
        tooltip=[alt.Tooltip("Date:T", title="Date"), alt.Tooltip("Concentration:Q", title="Concentration")],
    )

    line = base.mark_line()
    points = base.mark_point(size=40)
    st.altair_chart((line + points).properties(height=height), use_container_width=True)


def value_to_hex_color(v: float, vmin: float, vmax: float) -> str:
    if v is None or pd.isna(v):
        return "#888888"
    if vmax <= vmin:
        return "#ff0000"
    t = (v - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, float(t)))

    r = int(255 * t)
    g = 0
    b = int(255 * (1 - t))
    return f"#{r:02x}{g:02x}{b:02x}"


def add_gradient_legend(m: folium.Map, vmin: float, vmax: float, title: str):
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background: white;
        padding: 10px 12px;
        border: 1px solid #bbb;
        border-radius: 6px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        font-size: 12px;
        width: 240px;
    ">
      <div style="font-weight: 600; margin-bottom: 6px;">{title}</div>
      <div style="display:flex; align-items:center; gap:8px;">
        <span style="min-width:55px; text-align:left;">low</span>
        <div style="
          flex: 1;
          height: 12px;
          background: linear-gradient(to right, #0000ff, #ff0000);
          border: 1px solid #999;
        "></div>
        <span style="min-width:55px; text-align:right;">high</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:6px;">
        <span>{vmin:.4g}</span>
        <span>{vmax:.4g}</span>
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


# ----------------------------
# 0) Select Spatial Filter: HUC8 or County
# ----------------------------
st.header("0) Select Spatial Filter (Optional)")

filter_mode = st.radio(
    "Choose spatial filter type:",
    ["HUC8", "County"],
    horizontal=True
)

# default: nothing selected
selected_huc8s = []
selected_counties = []

if filter_mode == "HUC8":
    huc8_col, huc8_values = load_huc8_list(API_BASE)

    if not huc8_values:
        st.caption("No HUC8 values found (or HUC8 column not available). Showing all data.")
        selected_huc8s = []
    else:
        selected_huc8s = st.multiselect(
            "Choose one or more HUC8 basins (leave empty for ALL)",
            options=huc8_values,
            default=[],
        )

        if selected_huc8s:
            st.caption(
                f"Filtering all results to HUC8 in: {', '.join(selected_huc8s[:8])}"
                + (" ..." if len(selected_huc8s) > 8 else "")
            )
        else:
            st.caption("No HUC8 filter applied (showing ALL basins).")

else:
    county_col, county_values = load_county_list(API_BASE)

    if not county_values:
        st.caption("No County values found (or County column not available). Showing all data.")
        selected_counties = []
    else:
        # optional search box for faster filtering
        query = st.text_input("Search County (type part of name)", value="")
        if query.strip():
            county_values_filtered = [c for c in county_values if query.lower() in str(c).lower()]
        else:
            county_values_filtered = county_values

        selected_counties = st.multiselect(
            "Choose one or more Counties (leave empty for ALL)",
            options=county_values_filtered,
            default=[],
        )

        if selected_counties:
            st.caption(
                f"Filtering all results to County in: {', '.join(selected_counties[:8])}"
                + (" ..." if len(selected_counties) > 8 else "")
            )
        else:
            st.caption("No County filter applied (showing ALL counties).")

selected_huc8s_t = tuple(selected_huc8s)
selected_counties_t = tuple(selected_counties)
st.divider()


# ----------------------------
# 1) Select Parameter - MULTI
# ----------------------------
st.header("1) Select Parameter")

param_col, params = load_parameters(API_BASE, selected_huc8s_t, selected_counties_t)

if not params:
    st.error("No parameters found for the selected spatial filter (or /parameters not available).")
    st.stop()

selected_params = st.multiselect(
    "Choose one or more parameters",
    options=params,
    default=[params[0]] if params else [],
)

if param_col is None:
    st.caption("Single-parameter dataset (no parameter column detected).")
else:
    st.caption(f"Detected parameter column: `{param_col}`")

if not selected_params:
    st.warning("Please select at least one parameter to continue.")
    st.stop()

selected_params_t = tuple(selected_params)
plot_param = st.selectbox("Plot parameter (for charts)", options=list(selected_params), index=0)

# ✅ minimum samples per station
st.caption("Station filter")
min_samples_ui = st.number_input("Minimum samples per station", min_value=1, value=1, step=1)

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
            st.warning("No stations returned (check filters).")
        else:
            if "n_samples" in df_st.columns:
                st.caption("`n_samples` = number of rows for that station AFTER current filters.")
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
    st.warning("No stations returned for current filters. Try lowering min_samples, clearing filter, or changing parameters.")
    st.stop()

df_picker = df_picker.copy()

# label includes sample count if available
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
    "Select a station (search by typing)",
    options=df_picker["label"].tolist(),
    index=0,
)

station_id_input = selected_label.split("|")[0].strip()
st.caption(f"Selected station_id: **{station_id_input}**")

st.divider()

# ----------------------------
# 4) Station data -> CSV (+ chart + mean)
# ----------------------------
st.header("4) Station Data → CSV (with Time Series Plot)")

col1, col2 = st.columns(2)

with col1:
    if st.button("Fetch station data"):
        if not station_id_input:
            st.error("No station selected.")
        else:
            df_raw = load_station_data(API_BASE, station_id_input, selected_params_t, selected_huc8s_t, selected_counties_t)

            if df_raw.empty:
                st.warning("No data found for this station (given current filters).")
            else:
                st.success(f"Rows: {len(df_raw)}")

                df_plot_raw = df_raw.copy()
                if param_col and param_col in df_plot_raw.columns:
                    df_plot_raw = df_plot_raw[
                        df_plot_raw[param_col].astype(str).str.strip() == str(plot_param).strip()
                    ]

                df_ts = parse_timeseries(df_plot_raw)
                if df_ts.empty:
                    st.warning("No valid data points for the selected plot parameter after parsing.")
                else:
                    mean_val = compute_mean_from_timeseries(df_ts)
                    st.markdown(f"**Mean:** {mean_val:.4g}" if mean_val is not None else "**Mean:** (not available)")
                    st.subheader(f"Concentration vs Time ({plot_param})")
                    render_timeseries_chart(df_ts, height=350)

                st.dataframe(df_raw, use_container_width=True)
                st.download_button(
                    "Download this station as CSV",
                    df_raw.to_csv(index=False).encode("utf-8"),
                    file_name=f"station_{station_id_input}.csv",
                    mime="text/csv",
                )

with col2:
    st.info("Station is selected from the picker above.")

# ----------------------------
# 5) Download by year range
# ----------------------------
st.header("5) Download by Year Range")
c1, c2, c3 = st.columns([1, 1, 2])
start_year = c1.number_input("Start year", value=2000, step=1)
end_year = c2.number_input("End year", value=2010, step=1)

if c3.button("Download range CSV from API"):
    params_dl = build_query_params(
        parameters=list(selected_params_t),
        huc8s=list(selected_huc8s_t),
        counties=list(selected_counties_t),
        start_year=int(start_year),
        end_year=int(end_year),
    )

    content = None
    last_exc = None
    for i in range(6):
        try:
            r = requests.get(f"{API_BASE}/download-range", params=params_dl, timeout=360)
            r.raise_for_status()
            content = r.content
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5 * (i + 1))
    if content is None:
        raise last_exc

    st.download_button(
        "Click to download subset.csv",
        content,
        file_name="subset.csv",
        mime="text/csv",
    )

# ----------------------------
# 6) Map + time series plot (+ mean)
# ----------------------------
st.header("6) Station Map + Time Series")

if "show_map" not in st.session_state:
    st.session_state["show_map"] = False
if "selected_station_id" not in st.session_state:
    st.session_state["selected_station_id"] = ""

btn1, btn2 = st.columns(2)
with btn1:
    if st.button("Show map"):
        st.session_state["show_map"] = True
with btn2:
    if st.button("Hide map"):
        st.session_state["show_map"] = False

if st.session_state["show_map"]:
    map_col, plot_col = st.columns([1.2, 1])

    with map_col:
        with st.spinner("Loading stations and building map..."):
            df_map = load_stations_df(
                API_BASE,
                selected_params_t,
                selected_huc8s_t,
                selected_counties_t,
                max_points=5000,
                min_samples=min_samples_ui,
            )

            if df_map.empty:
                st.warning("No stations available (check filters or lower min_samples).")
            else:
                total_stations = len(df_map)
                st.markdown(f"**Stations shown on map (capped):** {total_stations}")
                st.caption("Map is capped to avoid heavy rendering. Use Station Picker for exact station selection.")

                center_lat = float(df_map["lat"].mean())
                center_lon = float(df_map["lon"].mean())
                m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

                for _, row in df_map.iterrows():
                    sid = str(row["MonitoringLocationIdentifierCor"])
                    n = None
                    if "n_samples" in df_map.columns:
                        try:
                            n = int(row.get("n_samples")) if pd.notna(row.get("n_samples")) else None
                        except Exception:
                            n = None

                    tooltip = f"{sid}" if n is None else f"{sid} (n={n})"
                    popup = f"<b>{sid}</b><br>lat={row['lat']}, lon={row['lon']}"
                    if n is not None:
                        popup += f"<br>n_samples={n}"

                    folium.CircleMarker(
                        location=[float(row["lat"]), float(row["lon"])],
                        radius=4,
                        tooltip=tooltip,
                        popup=popup,
                        fill=True,
                    ).add_to(m)

                folium_result = st_folium(m, width=900, height=550, key="station_map")

                clicked_sid = None
                if isinstance(folium_result, dict):
                    clicked_sid = folium_result.get("last_object_clicked_tooltip")

                if clicked_sid:
                    st.session_state["selected_station_id"] = str(clicked_sid).split(" (n=")[0].strip()

                if st.session_state["selected_station_id"]:
                    st.caption(f"Selected station_id (from map): {st.session_state['selected_station_id']}")
                else:
                    st.caption("Click a station marker to load its time series on the right.")

    with plot_col:
        st.subheader(f"Concentration vs Time ({plot_param})")

        sid = st.session_state["selected_station_id"]
        if not sid:
            st.info("Select a station on the map to display the time series.")
        else:
            with st.spinner(f"Fetching data for {sid}..."):
                df_raw = load_station_data(API_BASE, sid, selected_params_t, selected_huc8s_t, selected_counties_t)

            if df_raw.empty:
                st.warning("No data found for this station (given current filters).")
            else:
                df_plot_raw = df_raw.copy()
                if param_col and param_col in df_plot_raw.columns:
                    df_plot_raw = df_plot_raw[
                        df_plot_raw[param_col].astype(str).str.strip() == str(plot_param).strip()
                    ]

                df_ts = parse_timeseries(df_plot_raw)
                if df_ts.empty:
                    st.warning("No valid data points for the selected plot parameter after parsing.")
                else:
                    mean_val = compute_mean_from_timeseries(df_ts)
                    st.markdown(f"**Mean:** {mean_val:.4g}" if mean_val is not None else "**Mean:** (not available)")
                    render_timeseries_chart(df_ts, height=350)

                st.download_button(
                    "Download selected station as CSV",
                    df_raw.to_csv(index=False).encode("utf-8"),
                    file_name=f"station_{sid}.csv",
                    mime="text/csv",
                )

                with st.expander("Show raw table"):
                    st.dataframe(df_raw, use_container_width=True)

# ----------------------------
# 7) Mean Concentration Map
# ----------------------------
st.header("7) Mean Concentration Map (Blue = Low, Red = High)")

st.markdown(
    "This section creates a separate map for each selected parameter. "
    "Stations are colored by **mean concentration** at that station."
)

if "show_mean_maps" not in st.session_state:
    st.session_state["show_mean_maps"] = False

cbtn1, cbtn2 = st.columns(2)
with cbtn1:
    if st.button("Show mean maps"):
        st.session_state["show_mean_maps"] = True
with cbtn2:
    if st.button("Hide mean maps"):
        st.session_state["show_mean_maps"] = False

if st.session_state["show_mean_maps"]:
    df_coords = load_stations_df(
        API_BASE,
        selected_params_t,
        selected_huc8s_t,
        selected_counties_t,
        max_points=20000,
        min_samples=min_samples_ui,
    )
    if df_coords.empty:
        st.warning("No station coordinates available for the current filters.")
        st.stop()

    for i, p in enumerate(selected_params_t):
        st.subheader(f"Mean Map: {p}")

        with st.spinner(f"Computing mean per station for {p}..."):
            df_mean = load_mean_per_station(API_BASE, p, selected_huc8s_t, selected_counties_t)

        if df_mean.empty:
            st.warning(f"No mean-per-station data returned for: {p}")
            continue

        df_mean["mean_value"] = pd.to_numeric(df_mean.get("mean_value"), errors="coerce")
        df_mean = df_mean.dropna(subset=["mean_value"])

        df_join = df_mean.merge(df_coords, on="MonitoringLocationIdentifierCor", how="inner")
        df_join["lat"] = pd.to_numeric(df_join["lat"], errors="coerce")
        df_join["lon"] = pd.to_numeric(df_join["lon"], errors="coerce")
        df_join = df_join.dropna(subset=["lat", "lon", "mean_value"])

        if df_join.empty:
            st.warning(f"No stations with both coordinates and mean values for: {p}")
            continue

        df_join = df_join.head(5000)

        vmin = float(df_join["mean_value"].min())
        vmax = float(df_join["mean_value"].max())

        st.markdown(f"**Stations on map (capped):** {len(df_join)}")
        st.caption(f"Mean range: {vmin:.4g} → {vmax:.4g}")

        center_lat = float(df_join["lat"].mean())
        center_lon = float(df_join["lon"].mean())
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

        add_gradient_legend(m, vmin, vmax, title=f"Mean Concentration ({p})")

        for _, row in df_join.iterrows():
            sid = str(row["MonitoringLocationIdentifierCor"])
            mv = float(row["mean_value"])
            color = value_to_hex_color(mv, vmin, vmax)

            folium.CircleMarker(
                location=[float(row["lat"]), float(row["lon"])],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.85,
                tooltip=sid,
                popup=f"<b>{sid}</b><br>mean={mv:.4g}<br>lat={row['lat']}, lon={row['lon']}",
            ).add_to(m)

        st_folium(m, width=1100, height=600, key=f"mean_map_{i}")