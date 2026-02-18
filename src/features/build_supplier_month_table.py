from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import PROCESSED_DIR, SUPPLIER_MONTH_PATH


def find_input_parquet(processed_dir: Path) -> Path:
    """
    Pick a parquet file from data/processed that looks like the raw spend table.
    Exclude files we generate in later steps.
    """
    exclude = {"supplier_month.parquet", "supplier_month_labeled.parquet"}
    candidates = sorted([p for p in processed_dir.glob("*.parquet") if p.name not in exclude])

    if not candidates:
        raise FileNotFoundError(
            f"No suitable input parquet found in {processed_dir.resolve()}. "
            f"Put your raw parquet there (any name is fine)."
        )

    # If there are multiple, prefer the largest file (usually the raw dataset)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def build_supplier_month(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = _ensure_datetime(df)

    # Month start timestamp for stable grouping
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()

    # For robustness: use abs amount for "volume" features, keep signed sum too
    df["amount_abs"] = df["amount"].abs()

    gcols = ["vendor_number", "month"]

    agg = df.groupby(gcols).agg(
        tx_count=("amount", "size"),
        spend_sum=("amount", "sum"),
        spend_abs_sum=("amount_abs", "sum"),
        spend_mean=("amount", "mean"),
        spend_std=("amount", "std"),
        spend_max=("amount", "max"),
        spend_min=("amount", "min"),
        unique_companies=("company_code", "nunique"),
        unique_categories=("category", "nunique"),
        compliance_status=("compliance_status", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
        vendor_country=("vendor_country", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
        vendor_type=("vendor_type", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
        is_preferred=("is_preferred", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
        is_managed=("is_managed", lambda s: s.mode().iat[0] if not s.mode().empty else pd.NA),
    ).reset_index()

    agg["spend_std"] = agg["spend_std"].fillna(0.0)
    return agg


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    in_path = find_input_parquet(PROCESSED_DIR)
    df = pd.read_parquet(in_path)

    # Basic guardrails
    for col in ["date", "amount", "vendor_number", "company_code", "category", "compliance_status"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column `{col}` in input parquet: {in_path.name}")

    sm = build_supplier_month(df)
    sm.to_parquet(SUPPLIER_MONTH_PATH, index=False)

    print("✅ Built supplier-month table")
    print(f"Input: {in_path.resolve()}")
    print(f"Saved: {SUPPLIER_MONTH_PATH.resolve()}")
    print(f"Shape: {sm.shape}")
    print("Months:", sm["month"].min(), "→", sm["month"].max())
    print(sm.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
