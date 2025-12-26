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
    Read header only (no data) -> tiny memory.
    """
    path = _local_csv_path()
    cols = list(pd.read_csv(path, nrows=0).columns)
    return cols


@lru_cache(maxsize=1)
def get_param_col() -> Optional[str]:
    return detect_param_col(get_columns())


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


def _read_chunks(chunksize: int = 200_000, usecols: Optional[List[str]] = None):
    path = _local_csv_path()
    return pd.read_csv(path, low_memory=False, chunksize=chunksize, usecols=usecols)


def _apply_filters(
    chunk: pd.DataFrame,
    parameter: Optional[Union[str, List[str]]],
    huc8: Optional[Union[str, List[str]]],
) -> pd.DataFrame:
    param_col = get_param_col()

    if huc8 is not None and HUC8_COL in chunk.columns:
        hucs = set(_normalize_to_list(huc8))
        if hucs:
            col = chunk[HUC8_COL].astype(str).str.strip()
            chunk = chunk[col.isin(hucs)]

    if parameter is not None and param_col is not None and param_col in chunk.columns:
        params = set(_normalize_to_list(parameter))
        if params:
            col = chunk[param_col].astype(str).str.strip()
            chunk = chunk[col.isin(params)]

    return chunk


app = FastAPI(title="Water Quality API")


@app.get("/")
def home():
    cols = get_columns()
    param_col = get_param_col()

    return {
        "message": "Water API running!",
        "csv_file_env": CSV_FILE,
        "csv_local_path": _local_csv_path(),
        "parameter_column": param_col,
        "date_column_present": DATE_COL in cols,
        "huc8_column_present": HUC8_COL in cols,
        "columns": cols[:50],  # preview
    }


@app.get("/huc8-list")
def huc8_list():
    cols = get_columns()
    if HUC8_COL not in cols:
        return {"huc8_column": None, "huc8_values": []}

    values = set()
    usecols = [HUC8_COL]

    for chunk in _read_chunks(usecols=usecols):
        s = (
            chunk[HUC8_COL]
            .dropna()
            .astype(str)
            .str.strip()
        )
        for v in s.tolist():
            if v:
                values.add(v)

    return {"huc8_column": HUC8_COL, "huc8_values": sorted(values)}


@app.get("/parameters")
def parameters(huc8: Optional[List[str]] = Query(default=None)):
    cols = get_columns()
    param_col = get_param_col()

    if param_col is None or param_col not in cols:
        return {"parameter_column": None, "parameters": ["(single-parameter dataset)"]}

    usecols = [param_col]
    if HUC8_COL in cols:
        usecols.append(HUC8_COL)

    values = set()
    for chunk in _read_chunks(usecols=usecols):
        chunk = _apply_filters(chunk, parameter=None, huc8=huc8)
        s = (
            chunk[param_col]
            .dropna()
            .astype(str)
            .str.strip()
        )
        for v in s.tolist():
            if v:
                values.add(v)

    return {"parameter_column": param_col, "parameters": sorted(values)}


@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    param_col = get_param_col()

    needed = {STATION_COL, VALUE_COL}
    if HUC8_COL in cols:
        needed.add(HUC8_COL)
    if param_col is not None and param_col in cols:
        needed.add(param_col)

    missing = [c for c in [STATION_COL, VALUE_COL] if c not in cols]
    if missing:
        return {"error": f"Missing required columns: {missing}"}

    sums_counts: Dict[str, Tuple[float, int]] = {}

    for chunk in _read_chunks(usecols=list(needed)):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        if STATION_COL not in chunk.columns or VALUE_COL not in chunk.columns:
            continue

        vals = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.assign(_val=vals).dropna(subset=[STATION_COL, "_val"])

        for sid, grp in chunk.groupby(STATION_COL)["_val"]:
            s = float(grp.sum())
            c = int(grp.count())
            if sid in sums_counts:
                prev_s, prev_c = sums_counts[sid]
                sums_counts[sid] = (prev_s + s, prev_c + c)
            else:
                sums_counts[sid] = (s, c)

    out = []
    for sid, (s, c) in sums_counts.items():
        if c > 0:
            out.append({"MonitoringLocationIdentifierCor": sid, "mean_value": s / c})

    return out


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

    param_col = get_param_col()

    needed = set(cols)  # return full rows (can be heavy)
    # But to reduce memory, we still stream and stop at limit.

    results = []
    count = 0

    for chunk in _read_chunks(usecols=None):
        if STATION_COL not in chunk.columns:
            continue

        sub = chunk[chunk[STATION_COL] == station_id].copy()
        if sub.empty:
            continue

        sub = _apply_filters(sub, parameter=parameter, huc8=huc8)

        if DATE_COL in sub.columns:
            sub[DATE_COL] = pd.to_datetime(sub[DATE_COL], errors="coerce")

        if VALUE_COL in sub.columns:
            sub[VALUE_COL] = pd.to_numeric(sub[VALUE_COL], errors="coerce")

        # convert to safe JSON-ish
        rows = sub.fillna("").astype(str).to_dict(orient="records")
        for r in rows:
            results.append(r)
            count += 1
            if count >= limit:
                return {
                    "station_id": station_id,
                    "returned": len(results),
                    "limit": limit,
                    "note": "limit reached; use smaller filters or increase limit carefully",
                    "data": results,
                    "parameter_column": param_col,
                }

    return {
        "station_id": station_id,
        "returned": len(results),
        "limit": limit,
        "data": results,
        "parameter_column": param_col,
    }


@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    param_col = get_param_col()

    needed = set(cols)
    if DATE_COL not in cols:
        return {"error": f"Missing required date column: {DATE_COL}"}

    out_path = "/tmp/subset.csv"
    first_write = True

    for chunk in _read_chunks(usecols=None):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        if DATE_COL not in chunk.columns:
            continue

        dt = pd.to_datetime(chunk[DATE_COL], errors="coerce")
        chunk = chunk.assign(_dt=dt).dropna(subset=["_dt"])
        yrs = chunk["_dt"].dt.year
        chunk = chunk[(yrs >= start_year) & (yrs <= end_year)].drop(columns=["_dt"])

        if chunk.empty:
            continue

        chunk.to_csv(out_path, index=False, mode="w" if first_write else "a", header=first_write)
        first_write = False

    if first_write:
        # nothing written
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)

    return FileResponse(out_path, filename="subset.csv")


@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    param_col = get_param_col()

    required = [STATION_COL, LAT_COL, LON_COL]
    missing = [c for c in required if c not in cols]
    if missing:
        return {"error": f"Missing required columns: {missing}"}

    needed = {STATION_COL, LAT_COL, LON_COL}
    if HUC8_COL in cols:
        needed.add(HUC8_COL)
    if param_col is not None and param_col in cols:
        needed.add(param_col)

    seen: Dict[str, Tuple[float, float]] = {}

    for chunk in _read_chunks(usecols=list(needed)):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8)

        if LAT_COL in chunk.columns:
            chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        if LON_COL in chunk.columns:
            chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")

        chunk = chunk.dropna(subset=[STATION_COL, LAT_COL, LON_COL])

        for _, row in chunk[[STATION_COL, LAT_COL, LON_COL]].drop_duplicates().iterrows():
            sid = str(row[STATION_COL])
            if sid not in seen:
                seen[sid] = (float(row[LAT_COL]), float(row[LON_COL]))

    return [
        {"MonitoringLocationIdentifierCor": sid, "lat": lat, "lon": lon}
        for sid, (lat, lon) in seen.items()
    ]
