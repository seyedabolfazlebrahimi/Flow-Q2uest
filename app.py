# filename: app.py
import os
import urllib.request
from functools import lru_cache
from typing import Optional, List, Union, Dict, Tuple

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")  # local path OR URL

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
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"


# ------------------------------
# Helpers
# ------------------------------
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
    If CSV_FILE is a URL, download once to /tmp and reuse.
    """
    if not _is_url(CSV_FILE):
        return CSV_FILE

    local_path = "/tmp/NWQP-DO.csv"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(CSV_FILE, local_path)
    return local_path


@lru_cache(maxsize=1)
def get_columns() -> List[str]:
    """
    Read header only (cheap).
    """
    path = _local_csv_path()
    return list(pd.read_csv(path, nrows=0).columns)


@lru_cache(maxsize=1)
def get_param_col() -> Optional[str]:
    return detect_param_col(get_columns())


def _normalize_to_list(x: Optional[Union[str, List[str]]]) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if i and str(i).strip()]
    s = str(x).strip()
    return [s] if s else []


def _read_chunks(
    chunksize: int = 200_000,
    usecols: Optional[List[str]] = None,
):
    path = _local_csv_path()
    return pd.read_csv(
        path,
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )


def _apply_filters(
    chunk: pd.DataFrame,
    parameter: Optional[Union[str, List[str]]],
    huc8: Optional[Union[str, List[str]]],
) -> pd.DataFrame:
    param_col = get_param_col()

    if huc8 and HUC8_COL in chunk.columns:
        hucs = set(_normalize_to_list(huc8))
        if hucs:
            chunk = chunk[
                chunk[HUC8_COL].astype(str).str.strip().isin(hucs)
            ]

    if parameter and param_col and param_col in chunk.columns:
        params = set(_normalize_to_list(parameter))
        if params:
            chunk = chunk[
                chunk[param_col].astype(str).str.strip().isin(params)
            ]

    return chunk


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Water Quality API")


# Health checks (Render uses HEAD sometimes)
@app.head("/")
def home_head():
    return


@app.get("/")
def home():
    cols = get_columns()
    return {
        "message": "Water API running!",
        "csv_file_env": CSV_FILE,
        "csv_local_path": _local_csv_path(),
        "parameter_column": get_param_col(),
        "huc8_column_present": HUC8_COL in cols,
        "date_column_present": DATE_COL in cols,
    }


# ------------------------------
# HUC8 list
# ------------------------------
@app.get("/huc8-list")
def huc8_list():
    cols = get_columns()
    if HUC8_COL not in cols:
        return {"huc8_column": None, "huc8_values": []}

    values = set()
    for chunk in _read_chunks(usecols=[HUC8_COL]):
        s = (
            chunk[HUC8_COL]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values.update(v for v in s.tolist() if v)

    return {"huc8_column": HUC8_COL, "huc8_values": sorted(values)}


# ------------------------------
# Parameters (FIXED for 502)
# ------------------------------
@app.get("/parameters")
def parameters(huc8: Optional[List[str]] = Query(default=None)):
    cols = get_columns()
    param_col = get_param_col()

    if not param_col or param_col not in cols:
        return {"parameter_column": None, "parameters": []}

    usecols = [param_col]
    if HUC8_COL in cols:
        usecols.append(HUC8_COL)

    values = set()

    for chunk in _read_chunks(usecols=usecols):
        # IMPORTANT: only filter HUC8 here
        chunk = _apply_filters(chunk, parameter=None, huc8=huc8)

        s = (
            chunk[param_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        values.update(v for v in s.tolist() if v)

    return {
        "parameter_column": param_col,
        "parameters": sorted(values),
    }


# ------------------------------
# Mean per station
# ------------------------------
@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    param_col = get_param_col()

    required = [STATION_COL, VALUE_COL]
    for c in required:
        if c not in cols:
            return {"error": f"Missing required column: {c}"}

    needed = {STATION_COL, VALUE_COL}
    if HUC8_COL in cols:
        needed.add(HUC8_COL)
    if param_col and param_col in cols:
        needed.add(param_col)

    sums_counts: Dict[str, Tuple[float, int]] = {}

    for chunk in _read_chunks(usecols=list(needed)):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        vals = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.assign(_val=vals).dropna(subset=[STATION_COL, "_val"])

        for sid, grp in chunk.groupby(STATION_COL)["_val"]:
            s = float(grp.sum())
            c = int(grp.count())
            ps, pc = sums_counts.get(sid, (0.0, 0))
            sums_counts[sid] = (ps + s, pc + c)

    return [
        {"MonitoringLocationIdentifierCor": sid, "mean_value": s / c}
        for sid, (s, c) in sums_counts.items()
        if c > 0
    ]


# ------------------------------
# Station data (safe)
# ------------------------------
@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    limit: int = 5000,
):
    cols = get_columns()
    if STATION_COL not in cols:
        return {"error": f"Missing required column: {STATION_COL}"}

    results = []
    count = 0

    for chunk in _read_chunks():
        if STATION_COL not in chunk.columns:
            continue

        sub = chunk[chunk[STATION_COL] == station_id]
        if sub.empty:
            continue

        sub = _apply_filters(sub, parameter=parameter, huc8=huc8)

        if DATE_COL in sub.columns:
            sub[DATE_COL] = pd.to_datetime(sub[DATE_COL], errors="coerce")
        if VALUE_COL in sub.columns:
            sub[VALUE_COL] = pd.to_numeric(sub[VALUE_COL], errors="coerce")

        for r in sub.fillna("").astype(str).to_dict(orient="records"):
            results.append(r)
            count += 1
            if count >= limit:
                break
        if count >= limit:
            break

    return {
        "station_id": station_id,
        "returned": len(results),
        "limit": limit,
        "data": results,
        "parameter_column": get_param_col(),
    }


# ------------------------------
# Download by year range
# ------------------------------
@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    if DATE_COL not in cols:
        return {"error": f"Missing required column: {DATE_COL}"}

    out_path = "/tmp/subset.csv"
    first = True

    for chunk in _read_chunks():
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        if DATE_COL not in chunk.columns:
            continue

        dt = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.assign(_dt=dt).dropna(subset=["_dt"])
        yrs = chunk["_dt"].dt.year
        chunk = chunk[(yrs >= start_year) & (yrs <= end_year)].drop(columns=["_dt"])

        if not chunk.empty:
            chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
            first = False

    if first:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)

    return FileResponse(out_path, filename="subset.csv")


# ------------------------------
# Stations (map)
# ------------------------------
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    required = [STATION_COL, LAT_COL, LON_COL]
    for c in required:
        if c not in cols:
            return {"error": f"Missing required column: {c}"}

    param_col = get_param_col()
    needed = {STATION_COL, LAT_COL, LON_COL}
    if HUC8_COL in cols:
        needed.add(HUC8_COL)
    if param_col and param_col in cols:
        needed.add(param_col)

    seen: Dict[str, Tuple[float, float]] = {}

    for chunk in _read_chunks(usecols=list(needed)):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")
        chunk = chunk.dropna(subset=[STATION_COL, LAT_COL, LON_COL])

        for _, r in chunk[[STATION_COL, LAT_COL, LON_COL]].drop_duplicates().iterrows():
            sid = str(r[STATION_COL])
            if sid not in seen:
                seen[sid] = (float(r[LAT_COL]), float(r[LON_COL]))

    return [
        {"MonitoringLocationIdentifierCor": sid, "lat": lat, "lon": lon}
        for sid, (lat, lon) in seen.items()
    ]
