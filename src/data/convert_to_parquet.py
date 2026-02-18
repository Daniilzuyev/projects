from __future__ import annotations

import pandas as pd

from src.config import RAW_XLSX_PATH, RAW_PARQUET_PATH

# Columns that should always be treated as string (preserve leading zeros, hyphens)
FORCE_STR_SUFFIXES = ("_code", "_id", "_number", "_no", "_doc", "_key")
FORCE_STR_CONTAINS = ("account", "gl", "vendor", "supplier", "po", "invoice", "document")

def should_force_string(col: str) -> bool:
    c = col.lower()
    return c.endswith(FORCE_STR_SUFFIXES) or any(x in c for x in FORCE_STR_CONTAINS)

def main():
    RAW_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not RAW_XLSX_PATH.exists():
        raise FileNotFoundError(f"File {RAW_XLSX_PATH} does not exist")

    print(f"Reading: {RAW_XLSX_PATH.resolve()}")
    df = pd.read_excel(RAW_XLSX_PATH, engine="openpyxl")
    print("Loaded:", df.shape)

    # Normalize column names
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # 1) Force all object columns to pandas string (prevents pyarrow type errors)
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string")

    # 2) Additionally force "code/id/number" columns (even if pandas inferred numeric)
    forced = 0
    for c in df.columns:
        if should_force_string(c):
            df[c] = df[c].astype("string")
            forced += 1

    print(f"Coerced object cols to string: {len(obj_cols)}")
    print(f"Forced string by rule (codes/ids/etc): {forced}")

    df.to_parquet(RAW_PARQUET_PATH, engine="pyarrow", index=False)
    print("✅ Saved:", RAW_PARQUET_PATH.resolve())

if __name__ == "__main__":
    main()
