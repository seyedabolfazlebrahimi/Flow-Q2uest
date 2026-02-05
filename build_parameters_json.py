# build_huc8_list_json.py
import json
import pandas as pd

CSV_PATH = "NWQP-DO.csv"   # مسیر فایل csv روی لپتاپت
HUC8_COL = "HUC8"
CHUNK = 200_000

vals = set()
for chunk in pd.read_csv(CSV_PATH, usecols=[HUC8_COL], chunksize=CHUNK, low_memory=False):
    s = chunk[HUC8_COL].dropna().astype(str).str.strip()
    for v in s.tolist():
        if v:
            vals.add(v)

out = {"huc8_column": HUC8_COL, "huc8_values": sorted(vals)}
with open("huc8_list.json", "w", encoding="utf-8") as f:
    json.dump(out, f)

print("Saved huc8_list.json, count:", len(out["huc8_values"]))
