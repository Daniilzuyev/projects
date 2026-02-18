from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import FEATURES_PATH, SUPPLIER_MONTH_PATH
from src.features.pipeline import FeatureConfig, build_feature_table


def main() -> None:
    print("CWD:", Path.cwd())
    print("INPUT:", SUPPLIER_MONTH_PATH)
    print("INPUT exists:", SUPPLIER_MONTH_PATH.exists())

    df = pd.read_parquet(SUPPLIER_MONTH_PATH)
    df = build_feature_table(df, FeatureConfig())

    # Drop rows without sufficient history (first months)
    df = df.dropna(subset=["tx_count_lag1", "spend_sum_lag1"])

    df.to_parquet(FEATURES_PATH, index=False)

    print("✅ Feature table v1 created")
    print(f"Saved to: {FEATURES_PATH}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
