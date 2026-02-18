from __future__ import annotations

import pandas as pd


def normalize_vendor_number(v: object) -> str:
    return "" if v is None else str(v).strip()


def normalize_month(m: object) -> str:
    """Canonical month key: 'YYYY-MM'."""
    if m is None:
        return ""
    s = str(m).strip()
    ts = pd.to_datetime(s, errors="coerce")
    if pd.isna(ts):
        return s[:7]
    return ts.to_period("M").strftime("%Y-%m")


def month_series_to_yyyymm(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series).dt.to_period("M").dt.strftime("%Y-%m")
    return series.astype(str).apply(normalize_month)


def shift_month(yyyymm: str, months_delta: int) -> str:
    """Shift 'YYYY-MM' by months_delta."""
    ts = pd.Period(yyyymm, freq="M").to_timestamp()
    ts2 = ts + pd.DateOffset(months=months_delta)
    return ts2.to_period("M").strftime("%Y-%m")
