# filename: app.py
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse

# ------------------------------
# Load CSV
# ------------------------------
df = pd.read_csv("NWQP-DO.csv", low_memory=False)

# Convert date
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Convert numeric columns that MATTER
df["CuratedResultMeasureValue"] = pd.to_numeric(df["CuratedResultMeasureValue"], errors="coerce")
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

app = FastAPI(title="Water Quality API")


@app.get("/")
def home():
    return {"message": "Water API running!"}


# ----------------------------------------------------------
# 1) Mean per station
# ----------------------------------------------------------
@app.get("/mean-per-station")
def mean_per_station():
    mean_df = (
        df.groupby("MonitoringLocationIdentifierCor")["CuratedResultMeasureValue"]
        .mean()
        .reset_index()
        .rename(columns={"CuratedResultMeasureValue": "mean_value"})
    )
    return mean_df.to_dict(orient="records")


# ----------------------------------------------------------
# 2) Raw data for one station
# ----------------------------------------------------------
@app.get("/station-data")
def station_data(station_id: str):
    sub = df[df["MonitoringLocationIdentifierCor"] == station_id].copy()

    # Safest conversion (prevents JSON crash)
    return sub.fillna("").astype(str).to_dict(orient="records")


# ----------------------------------------------------------
# 3) Year filter
# ----------------------------------------------------------
@app.get("/download-range")
def download_range(start_year: int, end_year: int):
    sub = df[
        (df["Date"].dt.year >= start_year) &
        (df["Date"].dt.year <= end_year)
    ]
    out = "subset.csv"
    sub.to_csv(out, index=False)
    return FileResponse(out, filename="subset.csv")


# ----------------------------------------------------------
# 4) Stations list (map-ready)
# ----------------------------------------------------------
@app.get("/stations")
def stations():
    stations_df = df[["MonitoringLocationIdentifierCor", "lat", "lon"]].drop_duplicates()
    stations_df = stations_df.dropna()  # remove bad coordinates
    return stations_df.to_dict(orient="records")
