from __future__ import annotations

import pandas as pd

from src.config import RAW_PARQUET_PATH, VENDOR_LOOKUP_PATH

VENDOR_ID_COL = "vendor_number"
NAME_COL = "original_vendor_name"


def main() -> None:
    df = pd.read_parquet(RAW_PARQUET_PATH, columns=[VENDOR_ID_COL, NAME_COL])

    # Keep only valid pairs
    df = df.dropna(subset=[VENDOR_ID_COL, NAME_COL]).copy()

    # Ensure vendor_number is int-like
    df[VENDOR_ID_COL] = pd.to_numeric(df[VENDOR_ID_COL], errors="coerce")
    df = df.dropna(subset=[VENDOR_ID_COL]).copy()
    df[VENDOR_ID_COL] = df[VENDOR_ID_COL].astype(int)

    # Resolve duplicates: pick the most frequent company_name per vendor_number
    # (Robust if names vary slightly)
    lookup = (
        df.groupby(VENDOR_ID_COL)[NAME_COL]
        .agg(lambda s: s.value_counts(dropna=False).index[0])
        .reset_index()
        .rename(columns={NAME_COL: "vendor_name"})
        .sort_values(VENDOR_ID_COL)
        .reset_index(drop=True)
    )

    VENDOR_LOOKUP_PATH.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_parquet(VENDOR_LOOKUP_PATH, index=False)

    print("✅ vendor_lookup created")
    print("Saved:", VENDOR_LOOKUP_PATH)
    print("Rows:", len(lookup))
    print("Columns:", lookup.columns.tolist())
    print("Sample:")
    print(lookup.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
