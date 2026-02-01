# filename: app.py
import os
import urllib.request
from functools import lru_cache
from typing import Optional, List

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")

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
COUNTY_COL = "CountyName"          # ✅ ستون رسمی County
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"

DEFAULT_CHUNKSIZE = 200_000
LOCAL_CSV = "/tmp/NWQP-DO.csv"

# ------------------------------
# Helpers
# ------------------------------
def detect_param_col(columns):
    for c in CANDIDATE_PARAM_COLS:
        if c in columns:
            return c
    return None


def _normalize_to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if i]
    return [str(x).strip()]


def _is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


@lru_cache(maxsize=1)
def _local_csv_path():
    if not _is_url(CSV_FILE):
        return CSV_FILE

    if not os.path.exists(LOCAL_CSV):
        urllib.request.urlretrieve(CSV_FILE, LOCAL_CSV)

    return LOCAL_CSV


@lru_cache(maxsize=1)
def get_columns():
    return list(pd.read_csv(_local_csv_path(), nrows=0).columns)


@lru_cache(maxsize=1)
def get_param_col():
    return detect_param_col(get_columns())


def _read_chunks(chunksize=DEFAULT_CHUNKSIZE, usecols=None):
    return pd.read_csv(
        _local_csv_path(),
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )


# ---------- County normalization ----------
def _normalize_county_series(s: pd.Series) -> pd.Series:
    """
    Normalize CountyName values:
    - remove trailing ' County'
    - strip whitespace
    - title-case
    """
    return (
        s.astype(str)
        .str.replace(r"\s+County$", "", regex=True)
        .str.strip()
        .str.title()
    )


def _apply_filters(chunk, parameter=None, huc8=None, county=None):
    param_col = get_param_col()

    if huc8 and HUC8_COL in chunk.columns:
        chunk = chunk[
            chunk[HUC8_COL].astype(str).isin(_normalize_to_list(huc8))
        ]

    if county and COUNTY_COL in chunk.columns:
        county_norm = [c.strip().title() for c in _normalize_to_list(county)]
        chunk = chunk.assign(
            _county_norm=_normalize_county_series(chunk[COUNTY_COL])
        )
        chunk = chunk[chunk["_county_norm"].isin(county_norm)]
        chunk = chunk.drop(columns=["_county_norm"])

    if parameter and param_col and param_col in chunk.columns:
        chunk = chunk[
            chunk[param_col].astype(str).isin(_normalize_to_list(parameter))
        ]

    return chunk


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Water Quality API")

# ---------- Home (health check) ----------
@app.get("/")
def home():
    cols = get_columns()
    return {
        "status": "running",
        "csv_file_env": CSV_FILE,
        "csv_local_path": _local_csv_path(),
        "parameter_column": get_param_col(),
        "has_huc8": HUC8_COL in cols,
        "has_county": COUNTY_COL in cols,
        "n_columns": len(cols),
    }


# ---------- Debug ----------
@app.get("/__debug")
def debug():
    return {
        "csv_local_path": _local_csv_path(),
        "columns": get_columns(),
    }


# ---------- HUC8 ----------
@app.get("/huc8-list")
def huc8_list():
    if HUC8_COL not in get_columns():
        return {"huc8_column": None, "huc8_values": []}

    values = set()
    for chunk in _read_chunks(usecols=[HUC8_COL]):
        values.update(chunk[HUC8_COL].dropna().astype(str))

    return {"huc8_column": HUC8_COL, "huc8_values": sorted(values)}


# ---------- County ----------
@app.get("/county-list")
def county_list(
    q: Optional[str] = Query(
        default=None,
        description="Search string for CountyName (case-insensitive)",
    )
):
    if COUNTY_COL not in get_columns():
        return {"county_column": None, "counties": []}

    q_norm = q.strip().lower() if q else None
    values = set()

    for chunk in _read_chunks(usecols=[COUNTY_COL]):
        s = chunk[COUNTY_COL].dropna()
        if s.empty:
            continue

        s = _normalize_county_series(s)

        if q_norm:
            s = s[s.str.lower().str.contains(q_norm, na=False)]

        values.update(s.tolist())

    return {
        "county_column": COUNTY_COL,
        "counties": sorted(values),
        "search": q,
    }


# ---------- Parameters ----------
@app.get("/parameters")
def parameters(
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
):
    param_col = get_param_col()
    if not param_col:
        return {"parameter_column": None, "parameters": []}

    values = set()
    for chunk in _read_chunks(usecols=[param_col, HUC8_COL, COUNTY_COL]):
        chunk = _apply_filters(chunk, huc8=huc8, county=county)
        values.update(chunk[param_col].dropna().astype(str))

    return {"parameter_column": param_col, "parameters": sorted(values)}


# ---------- Stations ----------
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
    min_samples: int = 1,
):
    needed = [STATION_COL, LAT_COL, LON_COL, HUC8_COL, COUNTY_COL]
    agg = {}

    for chunk in _read_chunks(usecols=needed):
        chunk = _apply_filters(chunk, parameter, huc8, county)
        chunk = chunk.dropna(subset=[STATION_COL, LAT_COL, LON_COL])

        for sid, g in chunk.groupby(STATION_COL):
            lat = g[LAT_COL].iloc[0]
            lon = g[LON_COL].iloc[0]
            agg.setdefault(sid, [lat, lon, 0])
            agg[sid][2] += len(g)

    return [
        {
            "MonitoringLocationIdentifierCor": sid,
            "lat": lat,
            "lon": lon,
            "n_samples": n,
        }
        for sid, (lat, lon, n) in agg.items()
        if n >= min_samples
    ]


# ---------- Station data ----------
@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
    limit: int = 5000,
):
    results = []
    for chunk in _read_chunks():
        chunk = chunk[chunk[STATION_COL].astype(str) == str(station_id)]
        chunk = _apply_filters(chunk, parameter, huc8, county)
        results.extend(chunk.head(limit).to_dict(orient="records"))
        if len(results) >= limit:
            break

    return {"station_id": station_id, "data": results[:limit]}


# ---------- Download ----------
@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
):
    out = "/tmp/subset.csv"
    first = True

    for chunk in _read_chunks():
        chunk = _apply_filters(chunk, parameter, huc8, county)
        dt = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk[(dt.dt.year >= start_year) & (dt.dt.year <= end_year)]
        if not chunk.empty:
            chunk.to_csv(out, mode="w" if first else "a", index=False, header=first)
            first = False

    return FileResponse(out, filename="subset.csv")