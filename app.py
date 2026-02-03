# filename: app.py
import os
import json
import urllib.request
from functools import lru_cache
from typing import Optional, List, Union, Dict, Tuple, Iterable

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import duckdb

# ==============================
# DuckDB (read-only connection)
# ==============================
DB_FILE = "water_quality.duckdb"
con = duckdb.connect(DB_FILE, read_only=True)

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")
HUC8_INDEX_URL = (os.getenv("HUC8_INDEX_URL") or "").strip()
COUNTY_INDEX_URL = (os.getenv("COUNTY_INDEX_URL") or "").strip()

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
COUNTY_COL = "CountyName"
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"

INDEX_LOCAL_HUC8 = "/tmp/huc8_list.json"
INDEX_LOCAL_COUNTY = "/tmp/county_list.json"
DEFAULT_CHUNKSIZE = 200_000

# ------------------------------
# Helpers (CSV – موقت)
# ------------------------------
def detect_param_col(columns) -> Optional[str]:
    cols = set(columns)
    for c in CANDIDATE_PARAM_COLS:
        if c in cols:
            return c
    return None


@lru_cache(maxsize=1)
def _local_csv_path() -> str:
    return CSV_FILE


@lru_cache(maxsize=1)
def get_columns() -> List[str]:
    return list(pd.read_csv(_local_csv_path(), nrows=0).columns)


@lru_cache(maxsize=1)
def get_param_col() -> Optional[str]:
    return detect_param_col(get_columns())


def _read_chunks(
    chunksize: int = DEFAULT_CHUNKSIZE,
    usecols: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        _local_csv_path(),
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )


def _apply_filters(chunk: pd.DataFrame, parameter=None, huc8=None, county=None) -> pd.DataFrame:
    param_col = get_param_col()

    if huc8 and HUC8_COL in chunk.columns:
        chunk = chunk[chunk[HUC8_COL].astype(str).isin(huc8)]

    if county and COUNTY_COL in chunk.columns:
        chunk = chunk[chunk[COUNTY_COL].astype(str).isin(county)]

    if parameter and param_col and param_col in chunk.columns:
        chunk = chunk[chunk[param_col].astype(str).isin(parameter)]

    return chunk


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Water Quality API")


# ------------------------------
# DuckDB sanity check
# ------------------------------
@app.get("/_db-test")
def db_test():
    n = con.execute("SELECT COUNT(*) FROM wq").fetchone()[0]
    return {"rows_in_db": n}


# ======================================================
# 🚀 FAST — DuckDB-based /stations
# ======================================================
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
    min_samples: int = 1,
    max_points: int = 20000,
):
    where = []
    params: List = []

    param_col = get_param_col()

    if parameter and param_col:
        where.append(f"{param_col} IN ({','.join(['?'] * len(parameter))})")
        params.extend(parameter)

    if huc8:
        where.append(f"{HUC8_COL} IN ({','.join(['?'] * len(huc8))})")
        params.extend(huc8)

    if county:
        where.append(f"{COUNTY_COL} IN ({','.join(['?'] * len(county))})")
        params.extend(county)

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        SELECT
            {STATION_COL} AS MonitoringLocationIdentifierCor,
            CAST({LAT_COL} AS DOUBLE) AS lat,
            CAST({LON_COL} AS DOUBLE) AS lon,
            COUNT(*) AS n_samples
        FROM wq
        {where_sql}
        GROUP BY {STATION_COL}, lat, lon
        HAVING COUNT(*) >= ?
        LIMIT ?
    """

    params.extend([min_samples, max_points])
    df = con.execute(sql, params).df()
    return df.to_dict(orient="records")


# ======================================================
# 🚀 FAST — DuckDB-based /parameters
# ======================================================
@app.get("/parameters")
def parameters(
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
):
    param_col = get_param_col()
    if not param_col:
        return {"parameter_column": None, "parameters": []}

    where = []
    params: List = []

    if huc8:
        where.append(f"{HUC8_COL} IN ({','.join(['?'] * len(huc8))})")
        params.extend(huc8)

    if county:
        where.append(f"{COUNTY_COL} IN ({','.join(['?'] * len(county))})")
        params.extend(county)

    where_sql = "WHERE " + " AND ".join(where) if where else ""

    sql = f"""
        SELECT DISTINCT {param_col}
        FROM wq
        {where_sql}
        ORDER BY {param_col}
    """

    rows = con.execute(sql, params).fetchall()
    values = [r[0] for r in rows if r and r[0] is not None]

    return {
        "parameter_column": param_col,
        "parameters": values,
    }


# ======================================================
# 🚀 FAST — DuckDB-based /station-data   ✅ (مرحله ۶)
# ======================================================
@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
    limit: int = 5000,
):
    param_col = get_param_col()

    where = [f"{STATION_COL} = ?"]
    params: List = [station_id]

    if parameter and param_col:
        where.append(f"{param_col} IN ({','.join(['?'] * len(parameter))})")
        params.extend(parameter)

    if huc8:
        where.append(f"{HUC8_COL} IN ({','.join(['?'] * len(huc8))})")
        params.extend(huc8)

    if county:
        where.append(f"{COUNTY_COL} IN ({','.join(['?'] * len(county))})")
        params.extend(county)

    where_sql = "WHERE " + " AND ".join(where)

    sql = f"""
        SELECT *
        FROM wq
        {where_sql}
        ORDER BY {DATE_COL}
        LIMIT ?
    """

    params.append(int(limit))

    df = con.execute(sql, params).df()

    return {
        "station_id": station_id,
        "returned": len(df),
        "limit": int(limit),
        "data": df.replace({pd.NA: None}).to_dict(orient="records"),
        "parameter_column": param_col,
    }


# ======================================================
# ⏳ CSV-based — mean-per-station (مرحله بعد)
# ======================================================
@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
):
    cols = get_columns()
    param_col = get_param_col()

    if STATION_COL not in cols or VALUE_COL not in cols:
        return {"error": "Missing required columns"}

    needed = [STATION_COL, VALUE_COL]
    if HUC8_COL in cols:
        needed.append(HUC8_COL)
    if COUNTY_COL in cols:
        needed.append(COUNTY_COL)
    if param_col:
        needed.append(param_col)

    sums: Dict[str, Tuple[float, int]] = {}

    for chunk in _read_chunks(usecols=needed):
        chunk = _apply_filters(chunk, parameter, huc8, county)
        vals = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.assign(_v=vals).dropna(subset=[STATION_COL, "_v"])

        for sid, g in chunk.groupby(STATION_COL)["_v"]:
            s, c = g.sum(), g.count()
            ps, pc = sums.get(sid, (0.0, 0))
            sums[sid] = (ps + s, pc + c)

    return [
        {"MonitoringLocationIdentifierCor": sid, "mean_value": s / c}
        for sid, (s, c) in sums.items()
        if c > 0
    ]