# filename: app.py
import os
import io
from functools import lru_cache
from typing import Optional, List, Dict, Tuple, Iterable

import pandas as pd
import duckdb
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import StreamingResponse

# ==============================
# Config
# ==============================
DB_FILE = "water_quality.duckdb"
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
COUNTY_COL = "CountyName"
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"

DEFAULT_CHUNKSIZE = 200_000

# ==============================
# DuckDB connection (LAZY)
# ==============================
def get_db_connection():
    return duckdb.connect(DB_FILE, read_only=True)

# ==============================
# Helpers
# ==============================
def detect_param_col(columns) -> Optional[str]:
    for c in CANDIDATE_PARAM_COLS:
        if c in columns:
            return c
    return None


# ✅ NEW — Read columns from DuckDB instead of CSV
@lru_cache(maxsize=1)
def get_db_columns() -> List[str]:
    with get_db_connection() as con:
        rows = con.execute("PRAGMA table_info('wq')").fetchall()
    return [r[1] for r in rows]


# ✅ FIXED — Now uses DuckDB schema
@lru_cache(maxsize=1)
def get_param_col() -> Optional[str]:
    return detect_param_col(get_db_columns())


# CSV helpers (فقط برای mean-per-station)
@lru_cache(maxsize=1)
def get_columns() -> List[str]:
    return list(pd.read_csv(CSV_FILE, nrows=0).columns)


def _read_chunks(
    chunksize: int = DEFAULT_CHUNKSIZE,
    usecols: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        CSV_FILE,
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )


def _apply_filters(chunk, parameter=None, huc8=None, county=None):
    param_col = get_param_col()

    if huc8 and HUC8_COL in chunk.columns:
        chunk = chunk[chunk[HUC8_COL].astype(str).isin(huc8)]

    if county and COUNTY_COL in chunk.columns:
        chunk = chunk[chunk[COUNTY_COL].astype(str).isin(county)]

    if parameter and param_col and param_col in chunk.columns:
        chunk = chunk[chunk[param_col].astype(str).isin(parameter)]

    return chunk

# ==============================
# FastAPI app
# ==============================
app = FastAPI(title="Water Quality API")

# ==============================
# Root endpoint
# ==============================
@app.get("/")
def root():
    return {"status": "running", "service": "Water Quality API"}

# ==============================
# Sanity check
# ==============================
@app.get("/_db-test")
def db_test():
    with get_db_connection() as con:
        n = con.execute("SELECT COUNT(*) FROM wq").fetchone()[0]
    return {"rows_in_db": n}

# ==============================
# /huc8-list
# ==============================
@app.get("/huc8-list")
def huc8_list():
    with get_db_connection() as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT {HUC8_COL}
            FROM wq
            WHERE {HUC8_COL} IS NOT NULL
            ORDER BY {HUC8_COL}
            """
        ).fetchall()

    return {
        "huc8_column": HUC8_COL,
        "huc8_values": [r[0] for r in rows if r and r[0] is not None],
    }

# ==============================
# /county-list
# ==============================
@app.get("/county-list")
def county_list():
    with get_db_connection() as con:
        rows = con.execute(
            f"""
            SELECT DISTINCT {COUNTY_COL}
            FROM wq
            WHERE {COUNTY_COL} IS NOT NULL
            ORDER BY {COUNTY_COL}
            """
        ).fetchall()

    return {
        "county_column": COUNTY_COL,
        "county_values": [r[0] for r in rows if r and r[0] is not None],
    }

# ==============================
# /stations
# ==============================
@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),
    min_samples: int = 1,
    max_points: int = 20000,
):
    param_col = get_param_col()
    where = []
    params: List = []

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

    with get_db_connection() as con:
        df = con.execute(sql, params).df()

    return df.to_dict(orient="records")

# ==============================
# /parameters
# ==============================
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

    with get_db_connection() as con:
        rows = con.execute(sql, params).fetchall()

    return {
        "parameter_column": param_col,
        "parameters": [r[0] for r in rows if r and r[0] is not None],
    }

# ==============================
# /station-data
# ==============================
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

    sql = f"""
        SELECT *
        FROM wq
        WHERE {' AND '.join(where)}
        ORDER BY {DATE_COL}
        LIMIT ?
    """

    params.append(limit)

    with get_db_connection() as con:
        df = con.execute(sql, params).df()

    return {
        "station_id": station_id,
        "returned": len(df),
        "limit": limit,
        "data": df.replace({pd.NA: None}).to_dict(orient="records"),
        "parameter_column": param_col,
    }