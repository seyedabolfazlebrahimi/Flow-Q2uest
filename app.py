# filename: app.py
import os
import urllib.request
from functools import lru_cache
from typing import Optional, List, Union

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")  # can be local path OR URL

CANDIDATE_PARAM_COLS = [
    "CuratedConstituent",
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


def detect_param_col(columns) -> Optional[str]:
    cols = set(columns)
    for c in CANDIDATE_PARAM_COLS:
        if c in cols:
            return c
    return None


def _is_url(s: str) -> bool:
    s = (s or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


@lru_cache(maxsize=1)
def _local_csv_path() -> str:
    """
    If CSV_FILE is a URL, download it once to /tmp and reuse.
    If CSV_FILE is a local path, return it.
    """
    if not _is_url(CSV_FILE):
        return CSV_FILE

    # Render has writable /tmp; keep a stable name
    local_path = "/tmp/NWQP-DO.csv"

    if not os.path.exists(local_path):
        urllib.request.urlretrieve(CSV_FILE, local_path)

    return local_path


@lru_cache(maxsize=1)
def get_df() -> pd.DataFrame:
    """
    Lazy-load the dataset (only when first endpoint is called).
    Cached for the life of the process.
    """
    path = _local_csv_path()

    _df = pd.read_csv(path, low_memory=False)

    # Safe conversions
    if DATE_COL in _df.columns:
        _df[DATE_COL] = pd.to_datetime(_df[DATE_COL], errors="coerce")

    if "CuratedResultMeasureValue" in _df.columns:
        _df["CuratedResultMeasureValue"] = pd.to_numeric(
            _df["CuratedResultMeasureValue"], errors="coerce"
        )

    if "lat" in _df.columns:
        _df["lat"] = pd.to_numeric(_df["lat"], errors="coerce")

    if "lon" in _df.columns:
        _df["lon"] = pd.to_numeric(_df["lon"], errors="coerce")

    return _df


@lru_cache(maxsize=1)
def get_param_col() -> Optional[str]:
    df = get_df()
    return detect_param_col(df.columns)


app = FastAPI(title="Water Quality API")


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
    param_col = get_param_col()
    if param_col is None:
        return base_df

    params = _normalize_to_list(parameter)
    if not params:
        return base_df

    col = base_df[param_col].astype(str).str.strip()
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
# Home
# ----------------------------------------------------------
@app.get("/")
def home():
    df = get_df()
    param_col = get_param_col()

    return {
        "message": "Water API running!",
        "csv_file_env": CSV_FILE,
        "csv_local_path": _local_csv_path(),
        "parameter_column": param_col,
        "date_column": DATE_COL if DATE_COL in df.columns else None,
        "huc8_column_present": bool(HUC8_COL in df.columns),
        "rows": int(len(df)),
    }


# ----------------------------------------------------------
# List available HUC8 values
# ----------------------------------------------------------
@app.get("/huc8-list")
def huc8_list():
    df = get_df()

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
    df = get_df()
    param_col = get_param_col()

    sub = filter_by_huc8(df, huc8)

    if param_col is None:
        return {"parameter_column": None, "parameters": ["(single-parameter dataset)"]}

    params = (
        sub[param_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )
    params = sorted(params)
    return {"parameter_column": param_col, "parameters": params}


# ----------------------------------------------------------
# Mean per station (multi-parameter + multi-huc8)
# ----------------------------------------------------------
@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    df = get_df()

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
    df = get_df()

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
    df = get_df()

    sub = filter_by_huc8(df, huc8).copy()
    sub = filter_by_parameter(sub, parameter)

    if DATE_COL in sub.columns:
        sub = sub[
            (sub[DATE_COL].dt.year >= start_year) &
            (sub[DATE_COL].dt.year <= end_year)
        ]

    out = "/tmp/subset.csv"
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
    df = get_df()

    sub = filter_by_huc8(df, huc8)
    sub = filter_by_parameter(sub, parameter)

    stations_df = sub[["MonitoringLocationIdentifierCor", "lat", "lon"]].drop_duplicates()
    stations_df = stations_df.dropna(subset=["lat", "lon"])
    return stations_df.to_dict(orient="records")
