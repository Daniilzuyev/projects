from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


NUM_COLS = [
    "tx_count",
    "spend_sum",
    "spend_abs_sum",
]


@dataclass(frozen=True)
class FeatureConfig:
    rolling_window: int = 3


def prepare_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["vendor_number", "month"])

    # Compliance flags (current month only)
    status = df["compliance_status"].astype(str).str.lower()
    df["is_noncompliant_t"] = (status == "noncompliant").astype(int)
    df["is_unmanaged_t"] = (status == "unmanaged").astype(int)

    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUM_COLS:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = df.groupby("vendor_number")[col].shift(lag)
    return df


def add_deltas(df: pd.DataFrame) -> pd.DataFrame:
    df["tx_count_delta_1m"] = df["tx_count_lag1"] - df["tx_count_lag2"]
    df["spend_sum_delta_1m"] = df["spend_sum_lag1"] - df["spend_sum_lag2"]
    return df


def add_rolling(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    for col in NUM_COLS:
        roll = df.groupby("vendor_number")[col].shift(1).rolling(cfg.rolling_window)
        df[f"{col}_mean_{cfg.rolling_window}m"] = roll.mean()
        df[f"{col}_std_{cfg.rolling_window}m"] = roll.std()

        df[f"{col}_cv_{cfg.rolling_window}m"] = (
            df[f"{col}_std_{cfg.rolling_window}m"] /
            df[f"{col}_mean_{cfg.rolling_window}m"].replace(0, np.nan)
        )
    return df


def add_compliance_history(df: pd.DataFrame) -> pd.DataFrame:
    df["last_noncompliant_month"] = df["month"].where(df["is_noncompliant_t"] == 1)
    df["last_noncompliant_month"] = df.groupby("vendor_number")["last_noncompliant_month"].ffill()

    df["months_since_noncompliant"] = (df["month"] - df["last_noncompliant_month"]).dt.days // 30
    df["months_since_noncompliant"] = df["months_since_noncompliant"].fillna(999)

    df["noncompliant_last_3m"] = (
        df.groupby("vendor_number")["is_noncompliant_t"].shift(1).rolling(3).sum()
    )

    return df.drop(columns=["last_noncompliant_month"])


def build_feature_table(df: pd.DataFrame, cfg: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    df = prepare_base(df)
    df = add_lags(df)
    df = add_deltas(df)
    df = add_rolling(df, cfg)
    df = add_compliance_history(df)
    return df
