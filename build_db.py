import duckdb
import os

CSV_FILE = "NWQP-DO.csv"
DB_FILE = "water_quality.duckdb"

if os.path.exists(DB_FILE):
    print("DuckDB already exists, skipping build.")
    exit(0)

print("Building DuckDB database from CSV...")

con = duckdb.connect(DB_FILE)

con.execute(f"""
    CREATE TABLE wq AS
    SELECT *
    FROM read_csv_auto('{CSV_FILE}')
""")

# Optional but highly recommended indexes
con.execute("CREATE INDEX idx_station ON wq(MonitoringLocationIdentifierCor)")
con.execute("CREATE INDEX idx_param ON wq(CuratedConstituent)")
con.execute("CREATE INDEX idx_huc8 ON wq(HUC8)")

con.close()

print("DuckDB build complete.")