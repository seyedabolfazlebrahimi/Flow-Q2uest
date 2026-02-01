# filename: ui_app.py
import os
import time
import requests
import pandas as pd
import streamlit as st

# ----------------------------
# Config
# ----------------------------
API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="FLOW Q2UEST", layout="wide")
st.title("FLOW Q2UEST")

# ----------------------------
# Helpers
# ----------------------------
def get_json(url, params=None):
    for i in range(5):
        try:
            r = requests.get(url, params=params, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception:
            time.sleep(1.5 * (i + 1))
    return {}


def build_params(parameters=None, huc8s=None, counties=None):
    params = []
    if parameters:
        for p in parameters:
            params.append(("parameter", p))
    if huc8s:
        for h in huc8s:
            params.append(("huc8", h))
    if counties:
        for c in counties:
            params.append(("county", c))
    return params


# ----------------------------
# Spatial filter
# ----------------------------
st.header("0) Spatial Filter")

mode = st.radio(
    "Filter by:",
    ["HUC8", "County"],
    horizontal=True,
)

# ---- HUC8 mode ----
if mode == "HUC8":
    huc8_values = get_json(
        f"{API_BASE}/huc8-list"
    ).get("huc8_values", [])

    selected_huc8 = st.multiselect(
        "Select HUC8 basin(s)",
        options=huc8_values,
        default=[],
    )
    selected_county = []

# ---- County mode ----
else:
    county_search = st.text_input(
        "Search County (type a few letters)",
        value="",
        placeholder="e.g. ala, palm, brow",
    )

    county_params = {}
    if county_search.strip():
        county_params["q"] = county_search.strip()

    county_values = get_json(
        f"{API_BASE}/county-list",
        params=county_params,
    ).get("counties", [])

    selected_county = st.multiselect(
        "Select County / Counties",
        options=county_values,
        default=[],
    )
    selected_huc8 = []

st.divider()

# ----------------------------
# Parameter
# ----------------------------
params_payload = get_json(
    f"{API_BASE}/parameters",
    build_params(
        huc8s=selected_huc8,
        counties=selected_county,
    ),
)

params = params_payload.get("parameters", [])

if not params:
    st.warning("No parameters found for the selected spatial filter.")
    st.stop()

selected_param = st.selectbox(
    "Select parameter",
    options=params,
    index=0,
)

st.divider()

# ----------------------------
# Stations
# ----------------------------
st.header("Stations")

if st.button("Load stations"):
    data = get_json(
        f"{API_BASE}/stations",
        build_params(
            parameters=[selected_param],
            huc8s=selected_huc8,
            counties=selected_county,
        ),
    )

    if not data:
        st.warning("No stations returned for this selection.")
    else:
        df = pd.DataFrame(data)
        st.caption(f"Stations returned: {len(df)}")
        st.dataframe(df, use_container_width=True)