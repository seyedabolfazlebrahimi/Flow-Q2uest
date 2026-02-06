# filename: ui_app.py
import os
import json
import time

import numpy as np
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
# Visualization helpers
# ----------------------------
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
        <span style="min-width:55px;">low</span>
        <div style="
          flex:1;
          height:12px;
          background: linear-gradient(to right, #0000ff, #ff0000);
          border:1px solid #999;
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
# Time series helpers
# ----------------------------
def parse_timeseries(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Robust timeseries parser:
    - Finds a valid date column + value column safely
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
    """
    Altair time series chart using standardized columns from parse_timeseries()
    """
    if df_ts is None or df_ts.empty:
        st.info("No timeseries data to plot.")
        return

    chart_df = df_ts.rename(columns={"CuratedResultMeasureValue": "Concentration"}).copy()

    base = alt.Chart(chart_df).encode(
        x=alt.X("Date:T", axis=alt.Axis(title="Year", format="%Y", tickCount="year")),
        y=alt.Y("Concentration:Q", title="Concentration"),
        tooltip=[
            alt.Tooltip("Date:T", title="Date"),
            alt.Tooltip("Concentration:Q", title="Concentration"),
        ],
    )

    line = base.mark_line()
    points = base.mark_point(size=40)

    st.altair_chart((line + points).properties(height=height), use_container_width=True)

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
# ----------------------------
# 4) Station data → CSV (+ plots)
# ----------------------------
st.header("4) Station data → CSV (+ plots)")

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

            # --------------------------------------------------
            # Filter by selected parameter
            # --------------------------------------------------
            if param_col and param_col in df_raw.columns:
                df_plot = df_raw[
                    df_raw[param_col].astype(str).str.strip()
                    == str(plot_param).strip()
                ].copy()
            else:
                df_plot = df_raw.copy()

            # --------------------------------------------------
            # Parse Date, Concentration, Flow
            # --------------------------------------------------
            df_plot["Date"] = pd.to_datetime(
                df_plot.get("ActivityStartDate", df_plot.get("Date")),
                errors="coerce",
            )

            df_plot["C"] = pd.to_numeric(
                df_plot.get("CuratedResultMeasureValue", df_plot.get("Value")),
                errors="coerce",
            )

            if "CuratedQ" in df_plot.columns:
                df_plot["Q"] = pd.to_numeric(df_plot["CuratedQ"], errors="coerce")
            else:
                df_plot["Q"] = pd.NA

            df_plot = df_plot.dropna(subset=["Date", "C"]).sort_values("Date")

            # ==================================================
            # Mean values (C + Q) — colored
            # ==================================================
            if df_plot.empty:
                st.warning("No valid concentration data after filtering.")
            else:
                mean_c = df_plot["C"].mean()

                if df_plot["Q"].notna().any():
                    mean_q = df_plot["Q"].dropna().mean()

                    st.markdown(
                        f"""
                        <span style="color:#0072B2"><b>Mean concentration:</b> {mean_c:.4g} mg/L</span><br>
                        <span style="color:#D55E00"><b>Mean discharge:</b> {mean_q:.4g} m³/s</span>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<span style='color:#0072B2'><b>Mean concentration:</b> {mean_c:.4g} mg/L</span>",
                        unsafe_allow_html=True,
                    )

                # ==================================================
                # Dual-axis time series (C + Q) — NO MARKERS
                # ==================================================
                base = alt.Chart(df_plot).encode(
                    x=alt.X("Date:T", title="Date")
                )

                # ---- Concentration (left axis, blue)
                c_line = base.mark_line(
                    color="#0072B2",
                    strokeWidth=2,
                ).encode(
                    y=alt.Y(
                        "C:Q",
                        title="Concentration (mg/L)",
                        axis=alt.Axis(titleColor="#0072B2"),
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("C:Q", title="Concentration (mg/L)"),
                    ],
                )

                # ---- Discharge (right axis, dark orange)
                if df_plot["Q"].notna().any():
                    q_line = base.mark_line(
                        color="#D55E00",
                        strokeWidth=2,
                        opacity=0.85,
                    ).encode(
                        y=alt.Y(
                            "Q:Q",
                            title="Discharge (m³/s)",
                            axis=alt.Axis(titleColor="#D55E00"),
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Q:Q", title="Discharge (m³/s)"),
                        ],
                    )

                    ts_chart = alt.layer(
                        c_line,
                        q_line
                    ).resolve_scale(
                        y="independent"
                    )
                else:
                    ts_chart = c_line
                    st.caption("No discharge data available for this station.")

                st.altair_chart(ts_chart, use_container_width=True)

            # ==================================================
            # Q vs C (log–log) + POWER-LAW FIT + R²
            # ==================================================
            st.subheader("Concentration vs Flow (log–log)")

            if df_plot["Q"].notna().any():
                df_qc = df_plot.dropna(subset=["Q", "C"])
                df_qc = df_qc[(df_qc["Q"] > 0) & (df_qc["C"] > 0)]

                if df_qc.empty:
                    st.info("No observations with both concentration and discharge.")
                else:
                    df_qc = df_qc.assign(
                        logQ=np.log10(df_qc["Q"]),
                        logC=np.log10(df_qc["C"]),
                    )

                    x = df_qc["logQ"].values
                    y = df_qc["logC"].values

                    b, a = np.polyfit(x, y, 1)
                    y_hat = a + b * x

                    ss_res = np.sum((y - y_hat) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot

                    base_qc = alt.Chart(df_qc)

                    scatter = base_qc.mark_circle(
                        size=60,
                        opacity=0.6,
                        color="#0072B2",
                    ).encode(
                        x=alt.X(
                            "Q:Q",
                            scale=alt.Scale(type="log"),
                            title="Discharge Q (m³/s)",
                        ),
                        y=alt.Y(
                            "C:Q",
                            scale=alt.Scale(type="log"),
                            title="Concentration (mg/L)",
                        ),
                        tooltip=[
                            alt.Tooltip("Date:T", title="Date"),
                            alt.Tooltip("Q:Q", title="Q (m³/s)"),
                            alt.Tooltip("C:Q", title="Concentration (mg/L)"),
                        ],
                    )

                    fit = base_qc.transform_regression(
                        "logQ",
                        "logC",
                        method="linear",
                    ).transform_calculate(
                        Q="pow(10, datum.logQ)",
                        C="pow(10, datum.logC)",
                    ).mark_line(
                        color="red",
                        strokeWidth=2,
                        strokeDash=[6, 4],
                    ).encode(
                        x="Q:Q",
                        y="C:Q",
                    )

                    st.altair_chart(scatter + fit, use_container_width=True)

                    st.caption(
                        f"Power-law fit: C = a · Qᵇ  |  b = {b:.2f},  R² = {r2:.2f}"
                    )
            else:
                st.info("No discharge data available for Q–C analysis.")

            # ==================================================
            # Raw table + download
            # ==================================================
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

    # ---------- MAP ----------
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
            st.markdown(f"**Stations shown on map (capped):** {len(df_map)}")

            center_lat = float(df_map["lat"].mean())
            center_lon = float(df_map["lon"].mean())
            m = folium.Map(location=[center_lat, center_lon], zoom_start=7)

            for _, row in df_map.iterrows():
                sid = str(row["MonitoringLocationIdentifierCor"])

                n = None
                if "n_samples" in df_map.columns and pd.notna(row.get("n_samples")):
                    try:
                        n = int(row["n_samples"])
                    except Exception:
                        n = None

                tooltip = sid if n is None else f"{sid} (n={n})"

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

            folium_result = st_folium(
                m,
                width=900,
                height=550,
                key="station_map",
            )

            clicked_sid = None
            if isinstance(folium_result, dict):
                clicked_sid = folium_result.get("last_object_clicked_tooltip")

            if clicked_sid:
                # tooltip might be "ID (n=...)" → extract ID safely
                st.session_state["selected_station_id"] = (
                    str(clicked_sid).split(" (n=")[0].strip()
                )

            if st.session_state["selected_station_id"]:
                st.caption(
                    f"Selected station_id (from map): "
                    f"**{st.session_state['selected_station_id']}**"
                )
            else:
                st.caption("Hover to see sample count. Click a station to plot its time series.")

    # ---------- TIME SERIES ----------
    with plot_col:
        st.subheader(f"Concentration vs Time ({plot_param})")

        sid = st.session_state["selected_station_id"]
        if not sid:
            st.info("Select a station on the map to display the time series.")
        else:
            with st.spinner(f"Fetching data for {sid}..."):
                df_raw = load_station_data(
                    API_BASE,
                    sid,
                    selected_params_t,
                    selected_huc8s_t,
                    selected_counties_t,
                )

            if df_raw.empty:
                st.warning("No data found for this station.")
            else:
                df_plot_raw = df_raw.copy()
                if param_col and param_col in df_plot_raw.columns:
                    df_plot_raw = df_plot_raw[
                        df_plot_raw[param_col].astype(str).str.strip()
                        == str(plot_param).strip()
                    ]

                df_ts = parse_timeseries(df_plot_raw)

                if df_ts.empty:
                    st.warning("No valid data points after parsing.")
                else:
                    mean_val = compute_mean_from_timeseries(df_ts)
                    st.markdown(
                        f"**Mean:** {mean_val:.4g}"
                        if mean_val is not None
                        else "**Mean:** (not available)"
                    )
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
    # ---- station coordinates (respect HUC8 / County / min_samples)
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
            df_mean = load_mean_per_station(
                API_BASE,
                p,
                selected_huc8s_t,
                selected_counties_t,
            )

        if df_mean.empty:
            st.warning(f"No mean-per-station data returned for: {p}")
            continue

        df_mean["mean_value"] = pd.to_numeric(
            df_mean.get("mean_value"), errors="coerce"
        )
        df_mean = df_mean.dropna(subset=["mean_value"])

        # ---- join mean values with coordinates
        df_join = df_mean.merge(
            df_coords,
            on="MonitoringLocationIdentifierCor",
            how="inner",
        )

        df_join["lat"] = pd.to_numeric(df_join["lat"], errors="coerce")
        df_join["lon"] = pd.to_numeric(df_join["lon"], errors="coerce")
        df_join = df_join.dropna(subset=["lat", "lon", "mean_value"])

        if df_join.empty:
            st.warning(f"No stations with both coordinates and mean values for: {p}")
            continue

        # cap for rendering performance
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
                popup=(
                    f"<b>{sid}</b>"
                    f"<br>mean={mv:.4g}"
                    f"<br>lat={row['lat']}, lon={row['lon']}"
                ),
            ).add_to(m)

        st_folium(
            m,
            width=1100,
            height=600,
            key=f"mean_map_{i}",
        )