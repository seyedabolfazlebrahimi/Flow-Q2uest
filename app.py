# filename: app.py
import os
import json
import urllib.request
from functools import lru_cache
from typing import Optional, List, Union, Dict, Tuple

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ==========================================================
# Config
# ==========================================================
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")   # local path OR URL
PARAM_JSON = "parameters_by_huc8.json"           # precomputed JSON

HUC8_COL = "HUC8"
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"

# ==========================================================
# JSON helpers (FAST)
# ==========================================================
@lru_cache(maxsize=1)
def load_param_json():
    with open(PARAM_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

# ==========================================================
# CSV helpers (SLOW – chunk-based)
# ==========================================================
def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")

@lru_cache(maxsize=1)
def _local_csv_path() -> str:
    if not _is_url(CSV_FILE):
        return CSV_FILE
    local = "/tmp/NWQP-DO.csv"
    if not os.path.exists(local):
        urllib.request.urlretrieve(CSV_FILE, local)
    return local

@lru_cache(maxsize=1)
def get_columns() -> List[str]:
    return list(pd.read_csv(_local_csv_path(), nrows=0).columns)

def _normalize_to_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if i and str(i).strip()]
    s = str(x).strip()
    return [s] if s else []

def _read_chunks(chunksize: int = 200_000, usecols=None):
    return pd.read_csv(
        _local_csv_path(),
        chunksize=chunksize,
        low_memory=False,
        usecols=usecols,
    )

def _apply_filters(chunk, parameter, huc8):
    if huc8 and HUC8_COL in chunk.columns:
        hucs = set(_normalize_to_list(huc8))
        chunk = chunk[chunk[HUC8_COL].astype(str).isin(hucs)]

    param_col = load_param_json()["parameter_column"]
    if parameter and param_col in chunk.columns:
        params = set(_normalize_to_list(parameter))
        chunk = chunk[chunk[param_col].astype(str).isin(params)]

    return chunk

# ==========================================================
# FastAPI app
# ==========================================================
app = FastAPI(title="Water Quality API")

@app.head("/")
def health_head():
    return

@app.get("/")
def home():
    meta = load_param_json()
    return {
        "message": "Water API running",
        "parameter_column": meta["parameter_column"],
        "huc8_count": len(meta["by_huc8"]),
        "total_parameters": len(meta["all_parameters"]),
    }

# ==========================================================
# FAST endpoints (JSON-based)
# ==========================================================
@app.get("/huc8-list")
def huc8_list():
    data = load_param_json()
    return {
        "huc8_column": HUC8_COL,
        "huc8_values": sorted(data["by_huc8"].keys()),
    }

@app.get("/parameters")
def parameters(huc8: Optional[List[str]] = Query(default=None)):
    data = load_param_json()

    if not huc8:
        return {
            "parameter_column": data["parameter_column"],
            "parameters": data["all_parameters"],
        }

    params = set()
    for h in huc8:
        params.update(data["by_huc8"].get(h, []))

    return {
        "parameter_column": data["parameter_column"],
        "parameters": sorted(params),
    }

# ==========================================================
# CSV-based endpoints (chunk-safe)
# ==========================================================
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    needed = [STATION_COL, LAT_COL, LON_COL, HUC8_COL]
    param_col = load_param_json()["parameter_column"]
    needed.append(param_col)

    seen = {}
    for chunk in _read_chunks(usecols=needed):
        chunk = _apply_filters(chunk, parameter, huc8)
        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")
        chunk = chunk.dropna(subset=[STATION_COL, LAT_COL, LON_COL])

        for _, r in chunk.iterrows():
            sid = str(r[STATION_COL])
            if sid not in seen:
                seen[sid] = (float(r[LAT_COL]), float(r[LON_COL]))

    return [
        {"MonitoringLocationIdentifierCor": k, "lat": v[0], "lon": v[1]}
        for k, v in seen.items()
    ]

@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    limit: int = 5000,
):
    results = []
    param_col = load_param_json()["parameter_column"]

    for chunk in _read_chunks():
        chunk = chunk[chunk[STATION_COL] == station_id]
        if chunk.empty:
            continue

        chunk = _apply_filters(chunk, parameter, huc8)

        for r in chunk.fillna("").astype(str).to_dict("records"):
            results.append(r)
            if len(results) >= limit:
                break
        if len(results) >= limit:
            break

    return {
        "station_id": station_id,
        "returned": len(results),
        "limit": limit,
        "data": results,
        "parameter_column": param_col,
    }

@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    sums = {}

    for chunk in _read_chunks():
        chunk = _apply_filters(chunk, parameter, huc8)
        chunk[VALUE_COL] = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[STATION_COL, VALUE_COL])

        for sid, g in chunk.groupby(STATION_COL)[VALUE_COL]:
            s, c = sums.get(sid, (0.0, 0))
            sums[sid] = (s + g.sum(), c + g.count())

    return [
        {"MonitoringLocationIdentifierCor": k, "mean_value": s / c}
        for k, (s, c) in sums.items()
        if c > 0
    ]

@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    out = "/tmp/subset.csv"
    first = True

    for chunk in _read_chunks():
        chunk = _apply_filters(chunk, parameter, huc8)
        chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.dropna(subset=[DATE_COL])
        yrs = chunk[DATE_COL].dt.year
        chunk = chunk[(yrs >= start_year) & (yrs <= end_year)]

        if not chunk.empty:
            chunk.to_csv(out, mode="w" if first else "a", index=False, header=first)
            first = False

    if first:
        pd.DataFrame().to_csv(out, index=False)

    return FileResponse(out, filename="subset.csv")
