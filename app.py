# filename: app.py
import os
from typing import Optional, List, Union

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")

CANDIDATE_PARAM_COLS = [
    "CuratedConstituent",  # your real parameter column
    "Parameter",
    "parameter",
    "CharacteristicName",
    "CharacteristicName_Cor",
    "CuratedCharacteristicName",
    "Characteristic",
    "Analyte",
    "ResultMeasureName",
]

HUC8_COL = "HUC8"
DATE_COL = "ActivityStartDate"  # ✅ Date column name

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv(CSV_FILE, low_memory=False)

# Safe conversions
if DATE_COL in df.columns:
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

if "CuratedResultMeasureValue" in df.columns:
    df["CuratedResultMeasureValue"] = pd.to_numeric(df["CuratedResultMeasureValue"], errors="coerce")

if "lat" in df.columns:
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

if "lon" in df.columns:
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")


def detect_param_col(columns) -> Optional[str]:
    cols = set(columns)
    for c in CANDIDATE_PARAM_COLS:
        if c in cols:
            return c
    return None


PARAM_COL = detect_param_col(df.columns)

app = FastAPI(title="Water Quality API")


@app.get("/")
def home():
    return {
        "message": "Water API running!",
        "csv_file": CSV_FILE,
        "parameter_column": PARAM_COL,
        "date_column": DATE_COL if DATE_COL in df.columns else None,
        "huc8_column_present": bool(HUC8_COL in df.columns),
        "rows": int(len(df)),
    }


# ----------------------------------------------------------
# Helpers: normalize inputs to list
# ----------------------------------------------------------
def _normalize_to_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = []
        for item in x:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    s = str(x).strip()
    return [s] if s else []


def filter_by_parameter(base_df: pd.DataFrame, parameter: Optional[Union[str, List[str]]]) -> pd.DataFrame:
    if PARAM_COL is None:
        return base_df

    params = _normalize_to_list(parameter)
    if not params:
        return base_df

    col = base_df[PARAM_COL].astype(str).str.strip()
    return base_df[col.isin(params)]


def filter_by_huc8(base_df: pd.DataFrame, huc8: Optional[Union[str, List[str]]]) -> pd.DataFrame:
    if HUC8_COL not in base_df.columns:
        return base_df

    hucs = _normalize_to_list(huc8)
    if not hucs:
        return base_df

    col = base_df[HUC8_COL].astype(str).str.strip()
    return base_df[col.isin(hucs)]


# ----------------------------------------------------------
# List available HUC8 values
# ----------------------------------------------------------
@app.get("/huc8-list")
def huc8_list():
    if HUC8_COL not in df.columns:
        return {"huc8_column": None, "huc8_values": []}

    vals = (
        df[HUC8_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    vals = sorted(vals)
    return {"huc8_column": HUC8_COL, "huc8_values": vals}


# ----------------------------------------------------------
# Parameters endpoint (can be filtered by multiple HUC8s)
# ----------------------------------------------------------
@app.get("/parameters")
def parameters(huc8: Optional[List[str]] = Query(default=None)):
    sub = filter_by_huc8(df, huc8)

    if PARAM_COL is None:
        return {"parameter_column": None, "parameters": ["(single-parameter dataset)"]}

    params = (
        sub[PARAM_COL]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    params = sorted(params)
    return {"parameter_column": PARAM_COL, "parameters": params}


# ----------------------------------------------------------
# Mean per station (multi-parameter + multi-huc8)
# ----------------------------------------------------------
@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    sub = filter_by_huc8(df, huc8)
    sub = filter_by_parameter(sub, parameter)

    mean_df = (
        sub.groupby("MonitoringLocationIdentifierCor")["CuratedResultMeasureValue"]
        .mean()
        .reset_index()
        .rename(columns={"CuratedResultMeasureValue": "mean_value"})
    )
    return mean_df.to_dict(orient="records")


# ----------------------------------------------------------
# Raw data for one station (multi-parameter + multi-huc8)
# ----------------------------------------------------------
@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    sub = df[df["MonitoringLocationIdentifierCor"] == station_id].copy()
    sub = filter_by_huc8(sub, huc8)
    sub = filter_by_parameter(sub, parameter)

    return sub.fillna("").astype(str).to_dict(orient="records")


# ----------------------------------------------------------
# Year range download (multi-parameter + multi-huc8)
# ----------------------------------------------------------
@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    sub = filter_by_huc8(df, huc8).copy()
    sub = filter_by_parameter(sub, parameter)

    if DATE_COL in sub.columns:
        sub = sub[
            (sub[DATE_COL].dt.year >= start_year) &
            (sub[DATE_COL].dt.year <= end_year)
        ]

    out = "subset.csv"
    sub.to_csv(out, index=False)
    return FileResponse(out, filename="subset.csv")


# ----------------------------------------------------------
# Stations list for map (multi-parameter + multi-huc8)
# ----------------------------------------------------------
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    sub = filter_by_huc8(df, huc8)
    sub = filter_by_parameter(sub, parameter)

    stations_df = sub[["MonitoringLocationIdentifierCor", "lat", "lon"]].drop_duplicates()
    stations_df = stations_df.dropna(subset=["lat", "lon"])
    return stations_df.to_dict(orient="records")
