from __future__ import annotations
import pandas as pd

from src.config import RAW_PARQUET_PATH

REQUIRED_COLUMNS = [
    "date",
    "amount",
    "vendor_number",
    "normalized_vendor_name",
    "company_code",
    "category",
    "compliance_status",
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise TypeError("`date` column must be datetime")

    if df["vendor_number"].isna().mean() > 0.05:
        print("⚠️ Warning: vendor_number has >5% missing values")

    if (df["amount"] == 0).mean() > 0.2:
        print("⚠️ Warning: many zero-amount transactions detected")


def main() -> None:
    df = pd.read_parquet(RAW_PARQUET_PATH)
    validate_schema(df)
    print("✅ Schema validation passed")


if __name__ == "__main__":
    main()
