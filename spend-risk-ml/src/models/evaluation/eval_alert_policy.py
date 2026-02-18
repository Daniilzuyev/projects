from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb

from src.config import ARTIFACT_PATH, FEATURES_PATH, LABELED_PATH

TARGET = "y_next_noncompliant"


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    order = np.argsort(-scores)
    topk = y_true[order][:k]
    return float(topk.mean()) if k > 0 else np.nan


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    order = np.argsort(-scores)
    topk = y_true[order][:k]
    pos = y_true.sum()
    if pos == 0:
        return np.nan
    return float(topk.sum() / pos)


def main():
    artifact = json.loads(Path(ARTIFACT_PATH).read_text())
    model_path = artifact.get("model_file") or artifact.get("model_path")
    feature_names = artifact.get("features_used") or artifact.get("feature_names")
    train_end = pd.to_datetime(artifact["data_window"]["train_end"])

    labeled = pd.read_parquet(LABELED_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    df = labeled.merge(
        feats,
        on=["vendor_number", "month"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feat"),
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["month", "vendor_number"]).reset_index(drop=True)

    test = df[df["month"] > train_end].copy()

    # Build X строго по schema artifact
    X = test.copy()
    y = X[TARGET].astype(int).to_numpy()
    X = X.drop(columns=[TARGET, "vendor_number", "month", "compliance_status"], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if c.endswith("_feat")], errors="ignore")
    X = X.select_dtypes(include=[np.number, "bool"])

    # Align columns: add missing=0, drop extra, reorder
    for c in feature_names:
        if c not in X.columns:
            X[c] = 0
    X = X[feature_names]

    if not model_path or not feature_names:
        raise RuntimeError("Artifact missing model path or feature list.")

    model = lgb.Booster(model_file=str(model_path))
    scores = model.predict(X)

    test["score"] = scores

    print("=== Test window ===")
    print("Months:", sorted(test["month"].astype(str).unique().tolist()))
    print("Rows:", len(test), "Positives:", int(y.sum()), "Rate:", float(y.mean()))

    print("\n=== Alert policy table ===")
    for k in [10, 25, 50, 75, 100, 150, 200, 250, 300]:
        p = precision_at_k(y, scores, k)
        r = recall_at_k(y, scores, k)
        print(f"K={k:>3} | Precision@K={p:.4f} | Recall@K={r:.3f}")

    # Show top-25 vendors for product demo
    topn = 25
    top = test.sort_values("score", ascending=False).head(topn)[
        ["vendor_number", "month", "score", TARGET, "compliance_status"]
    ]
    print(f"\n=== Top {topn} candidates (for demo) ===")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
