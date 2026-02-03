# filename: build_db.py
import duckdb
import os
import urllib.request

CSV_FILE = os.getenv("CSV_FILE", "NWQP-DO.csv")
DB_FILE = "water_quality.duckdb"

if os.path.exists(DB_FILE):
    print("DuckDB already exists, skipping build.")
    exit(0)

# اگر CSV URL بود، دانلودش کن
if CSV_FILE.startswith("http://") or CSV_FILE.startswith("https://"):
    local_csv = "data.csv"
    print(f"Downloading CSV from {CSV_FILE} ...")
    urllib.request.urlretrieve(CSV_FILE, local_csv)
else:
    local_csv = CSV_FILE

if not os.path.exists(local_csv):
    raise FileNotFoundError(f"CSV file not found: {local_csv}")

print("Building DuckDB database from CSV...")

con = duckdb.connect(DB_FILE)

con.execute(f"""
    CREATE TABLE wq AS
    SELECT *
    FROM read_csv_auto('{local_csv}')
""")

# Indexes (خیلی مهم برای سرعت)
con.execute("CREATE INDEX idx_station ON wq(MonitoringLocationIdentifierCor)")
con.execute("CREATE INDEX idx_param ON wq(CuratedConstituent)")
con.execute("CREATE INDEX idx_huc8 ON wq(HUC8)")

con.close()

print("DuckDB build complete.")