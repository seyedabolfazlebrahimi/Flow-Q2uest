# filename: app.py
import os
import json
import urllib.request
from functools import lru_cache
from typing import Optional, List, Union, Dict, Tuple, Iterable

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

# ------------------------------
# Config
# ------------------------------
CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")  # local path OR URL
HUC8_INDEX_URL = (os.getenv("HUC8_INDEX_URL") or "").strip()      # optional: precomputed HUC8 list
COUNTY_INDEX_URL = (os.getenv("COUNTY_INDEX_URL") or "").strip()  # optional: precomputed County list

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
COUNTY_COL = "CountyName"  # ✅ NEW
DATE_COL = "ActivityStartDate"
VALUE_COL = "CuratedResultMeasureValue"
STATION_COL = "MonitoringLocationIdentifierCor"
LAT_COL = "lat"
LON_COL = "lon"

INDEX_LOCAL_HUC8 = "/tmp/huc8_list.json"
INDEX_LOCAL_COUNTY = "/tmp/county_list.json"  # ✅ NEW

DEFAULT_CHUNKSIZE = 200_000


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
    """If CSV_FILE is a URL, download once to /tmp and reuse."""
    if not _is_url(CSV_FILE):
        return CSV_FILE

    local_path = "/tmp/NWQP-DO.csv"
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(CSV_FILE, local_path)
    return local_path


@lru_cache(maxsize=1)
def get_columns() -> List[str]:
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
    chunksize: int = DEFAULT_CHUNKSIZE,
    usecols: Optional[List[str]] = None,
) -> Iterable[pd.DataFrame]:
    path = _local_csv_path()
    return pd.read_csv(
        path,
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols,
    )


def _apply_filters(chunk: pd.DataFrame, parameter=None, huc8=None, county=None) -> pd.DataFrame:
    """
    Applies filters if the relevant columns exist in chunk.
    parameter: Optional[List[str]] or str
    huc8: Optional[List[str]] or str
    county: Optional[List[str]] or str   ✅ NEW
    """
    param_col = get_param_col()

    # HUC8 filter (unchanged)
    if huc8 and HUC8_COL in chunk.columns:
        hucs = set(_normalize_to_list(huc8))
        if hucs:
            chunk = chunk[chunk[HUC8_COL].astype(str).str.strip().isin(hucs)]

    # County filter (new, additive)
    if county and COUNTY_COL in chunk.columns:
        counties = set(_normalize_to_list(county))
        if counties:
            chunk = chunk[chunk[COUNTY_COL].astype(str).str.strip().isin(counties)]

    # Parameter filter (unchanged)
    if parameter and param_col and param_col in chunk.columns:
        params = set(_normalize_to_list(parameter))
        if params:
            chunk = chunk[chunk[param_col].astype(str).str.strip().isin(params)]

    return chunk


def _load_huc8_index():
    """
    Priority:
      1) HUC8_INDEX_URL (download once to /tmp)
      2) /tmp/huc8_list.json if exists
      3) None
    """
    if HUC8_INDEX_URL:
        if not os.path.exists(INDEX_LOCAL_HUC8):
            urllib.request.urlretrieve(HUC8_INDEX_URL, INDEX_LOCAL_HUC8)

    if os.path.exists(INDEX_LOCAL_HUC8):
        with open(INDEX_LOCAL_HUC8, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def _load_county_index():
    """
    Priority:
      1) COUNTY_INDEX_URL (download once to /tmp)
      2) /tmp/county_list.json if exists
      3) None
    """
    if COUNTY_INDEX_URL:
        if not os.path.exists(INDEX_LOCAL_COUNTY):
            urllib.request.urlretrieve(COUNTY_INDEX_URL, INDEX_LOCAL_COUNTY)

    if os.path.exists(INDEX_LOCAL_COUNTY):
        with open(INDEX_LOCAL_COUNTY, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI(title="Water Quality API")


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
        "county_column_present": COUNTY_COL in cols,  # ✅ NEW
        "date_column_present": DATE_COL in cols,
        "huc8_index_url_set": bool(HUC8_INDEX_URL),
        "huc8_index_local_exists": os.path.exists(INDEX_LOCAL_HUC8),
        "county_index_url_set": bool(COUNTY_INDEX_URL),  # ✅ NEW
        "county_index_local_exists": os.path.exists(INDEX_LOCAL_COUNTY),  # ✅ NEW
    }


@app.get("/huc8-list")
def huc8_list():
    # FAST PATH: precomputed index
    idx = _load_huc8_index()
    if isinstance(idx, dict) and "huc8_values" in idx:
        return {
            "huc8_column": idx.get("huc8_column", HUC8_COL),
            "huc8_values": idx.get("huc8_values", []),
            "source": "index",
        }

    cols = get_columns()
    if HUC8_COL not in cols:
        return {"huc8_column": None, "huc8_values": [], "source": "csv_scan"}

    values = set()
    for chunk in _read_chunks(usecols=[HUC8_COL]):
        s = chunk[HUC8_COL].dropna().astype(str).str.strip()
        values.update(v for v in s.tolist() if v)

    return {"huc8_column": HUC8_COL, "huc8_values": sorted(values), "source": "csv_scan"}


@app.get("/county-list")
def county_list():
    # FAST PATH: precomputed index
    idx = _load_county_index()
    if isinstance(idx, dict) and "county_values" in idx:
        return {
            "county_column": idx.get("county_column", COUNTY_COL),
            "county_values": idx.get("county_values", []),
            "source": "index",
        }

    cols = get_columns()
    if COUNTY_COL not in cols:
        return {"county_column": None, "county_values": [], "source": "csv_scan"}

    values = set()
    for chunk in _read_chunks(usecols=[COUNTY_COL]):
        s = chunk[COUNTY_COL].dropna().astype(str).str.strip()
        values.update(v for v in s.tolist() if v)

    return {"county_column": COUNTY_COL, "county_values": sorted(values), "source": "csv_scan"}


@app.get("/parameters")
def parameters(
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),  # ✅ NEW
):
    cols = get_columns()
    param_col = get_param_col()

    if not param_col or param_col not in cols:
        return {"parameter_column": None, "parameters": []}

    usecols = [param_col]
    if HUC8_COL in cols:
        usecols.append(HUC8_COL)
    if COUNTY_COL in cols:
        usecols.append(COUNTY_COL)

    values = set()
    for chunk in _read_chunks(usecols=usecols):
        chunk = _apply_filters(chunk, parameter=None, huc8=huc8, county=county)
        s = chunk[param_col].dropna().astype(str).str.strip()
        values.update(v for v in s.tolist() if v)

    return {"parameter_column": param_col, "parameters": sorted(values)}


@app.get("/mean-per-station")
def mean_per_station(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),  # ✅ NEW
):
    cols = get_columns()
    param_col = get_param_col()

    for c in [STATION_COL, VALUE_COL]:
        if c not in cols:
            return {"error": f"Missing required column: {c}"}

    needed = {STATION_COL, VALUE_COL}
    if HUC8_COL in cols:
        needed.add(HUC8_COL)
    if COUNTY_COL in cols:
        needed.add(COUNTY_COL)
    if param_col and param_col in cols:
        needed.add(param_col)

    sums_counts: Dict[str, Tuple[float, int]] = {}

    for chunk in _read_chunks(usecols=list(needed)):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8, county=county)

        vals = pd.to_numeric(chunk[VALUE_COL], errors="coerce")
        chunk = chunk.assign(_val=vals).dropna(subset=[STATION_COL, "_val"])

        gb = chunk.groupby(STATION_COL)["_val"].agg(["sum", "count"])
        for sid, row in gb.iterrows():
            s = float(row["sum"])
            c = int(row["count"])
            ps, pc = sums_counts.get(sid, (0.0, 0))
            sums_counts[sid] = (ps + s, pc + c)

    return [
        {"MonitoringLocationIdentifierCor": sid, "mean_value": s / c}
        for sid, (s, c) in sums_counts.items()
        if c > 0
    ]


@app.get("/station-data")
def station_data(
    station_id: str,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),  # ✅ NEW
    limit: int = 5000,
):
    cols = get_columns()
    if STATION_COL not in cols:
        return {"error": f"Missing required column: {STATION_COL}"}

    param_col = get_param_col()

    # read only what we need (critical for RAM)
    usecols = [STATION_COL]
    for c in [DATE_COL, VALUE_COL, LAT_COL, LON_COL]:
        if c in cols:
            usecols.append(c)
    if HUC8_COL in cols:
        usecols.append(HUC8_COL)
    if COUNTY_COL in cols:
        usecols.append(COUNTY_COL)
    if param_col and param_col in cols:
        usecols.append(param_col)

    results = []
    count = 0

    for chunk in _read_chunks(usecols=list(dict.fromkeys(usecols))):
        if STATION_COL not in chunk.columns:
            continue

        sub = chunk[chunk[STATION_COL].astype(str) == str(station_id)]
        if sub.empty:
            continue

        sub = _apply_filters(sub, parameter=parameter, huc8=huc8, county=county)
        if sub.empty:
            continue

        if DATE_COL in sub.columns:
            sub[DATE_COL] = pd.to_datetime(sub[DATE_COL], errors="coerce")
        if VALUE_COL in sub.columns:
            sub[VALUE_COL] = pd.to_numeric(sub[VALUE_COL], errors="coerce")

        recs = sub.replace({pd.NA: None}).to_dict(orient="records")
        for r in recs:
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
        "parameter_column": param_col,
    }


@app.get("/download-range")
def download_range(
    start_year: int,
    end_year: int,
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),  # ✅ NEW
):
    cols = get_columns()
    if DATE_COL not in cols:
        return {"error": f"Missing required column: {DATE_COL}"}

    out_path = "/tmp/subset.csv"
    first = True

    for chunk in _read_chunks():
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8, county=county)

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


@app.get("/stations")
def stations(
    parameter: Optional[List[str]] = Query(default=None),
    huc8: Optional[List[str]] = Query(default=None),
    county: Optional[List[str]] = Query(default=None),  # ✅ NEW
    min_samples: int = 1,      # minimum number of rows per station (after filters)
    max_points: int = 20000,   # cap to avoid huge RAM usage
):
    cols = get_columns()
    for c in [STATION_COL, LAT_COL, LON_COL]:
        if c not in cols:
            return {"error": f"Missing required column: {c}"}

    if min_samples is None:
        min_samples = 1
    min_samples = max(1, int(min_samples))
    max_points = max(1, int(max_points))

    param_col = get_param_col()

    # read only needed columns
    needed = [STATION_COL, LAT_COL, LON_COL]
    if HUC8_COL in cols:
        needed.append(HUC8_COL)
    if COUNTY_COL in cols:
        needed.append(COUNTY_COL)
    if param_col and param_col in cols:
        needed.append(param_col)

    # station -> (lat, lon, n_samples)
    agg: Dict[str, Tuple[float, float, int]] = {}

    for chunk in _read_chunks(usecols=needed):
        chunk = _apply_filters(chunk, parameter=parameter, huc8=huc8, county=county)
        if chunk.empty:
            continue

        # clean coords
        chunk[LAT_COL] = pd.to_numeric(chunk[LAT_COL], errors="coerce")
        chunk[LON_COL] = pd.to_numeric(chunk[LON_COL], errors="coerce")
        chunk = chunk.dropna(subset=[STATION_COL, LAT_COL, LON_COL])
        if chunk.empty:
            continue

        # station ids as strings
        sids = chunk[STATION_COL].astype(str).str.strip()

        # counts in this chunk
        counts = sids.value_counts()

        # first lat/lon per station in this chunk
        first_xy = (
            chunk.assign(_sid=sids)[["_sid", LAT_COL, LON_COL]]
            .drop_duplicates(subset=["_sid"])
            .set_index("_sid")
        )

        for sid, c in counts.items():
            if sid not in first_xy.index:
                continue
            lat = float(first_xy.loc[sid, LAT_COL])
            lon = float(first_xy.loc[sid, LON_COL])

            if sid in agg:
                plat, plon, pc = agg[sid]
                agg[sid] = (plat, plon, pc + int(c))
            else:
                agg[sid] = (lat, lon, int(c))

        # optional early stop for memory/time:
        if len(agg) >= max_points and min_samples <= 1:
            break

    out = []
    for sid, (lat, lon, n) in agg.items():
        if n >= min_samples:
            out.append(
                {
                    "MonitoringLocationIdentifierCor": sid,
                    "lat": lat,
                    "lon": lon,
                    "n_samples": n,
                }
            )

    # If user asks for many points, still cap output size
    if len(out) > max_points:
        out = out[:max_points]

    return out