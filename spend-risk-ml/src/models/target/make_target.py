from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def time_split(df: pd.DataFrame):
    months = sorted(df["month"].unique())
    test_month = months[-1]

    train = df[df["month"] < test_month].copy()
    test = df[df["month"] == test_month].copy()
    return train, test


def baseline_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Simple business rule:
    - already noncompliant -> high risk
    - otherwise -> low risk
    """
    return np.where(df["compliance_status"] == "noncompliant", 0.9, 0.05)


def main():
    root = project_root()
    path = root / "data" / "processed" / "supplier_month_labeled.parquet"

    df = pd.read_parquet(path)

    train, test = time_split(df)

    y_test = test["y_next_noncompliant"].to_numpy()
    p_test = baseline_predict_proba(test)

    # Metrics suitable for extreme imbalance
    pr_auc = average_precision_score(y_test, p_test)
    roc_auc = roc_auc_score(y_test, p_test) if y_test.sum() > 0 else float("nan")

    print("✅ Baseline (rules-only) evaluation")
    print("Train rows:", len(train))
    print("Test rows:", len(test))
    print("Test month:", test["month"].iloc[0])
    print("Target rate (test):", y_test.mean())
    print("PR AUC (Average Precision):", round(pr_auc, 5))
    print("ROC AUC:", round(roc_auc, 5))


if __name__ == "__main__":
    main()
