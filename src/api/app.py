# src/api/app.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import (
    ARTIFACT_PATH,
    FEATURES_PATH,
    MODEL_PATH,
    REVIEW_LIMIT_DEFAULT,
    TOP_K_DEFAULT,
    VENDOR_LOOKUP_PATH,
)
from src.inference.explain import SupplierExplainer
from src.utils.normalize import (
    month_series_to_yyyymm,
    normalize_month,
    normalize_vendor_number,
    shift_month,
)

try:
    import lightgbm as lgb
except Exception:
    lgb = None


# =========================
# Loaders
# =========================
@lru_cache(maxsize=1)
def load_features_df() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    if "vendor_number" not in df.columns or "month" not in df.columns:
        raise RuntimeError("Features parquet must contain: vendor_number, month")
    df["vendor_number"] = df["vendor_number"].apply(normalize_vendor_number)
    df["_month"] = month_series_to_yyyymm(df["month"])
    return df


@lru_cache(maxsize=1)
def load_vendor_lookup() -> pd.DataFrame:
    df = pd.read_parquet(VENDOR_LOOKUP_PATH)
    if "vendor_number" not in df.columns or "vendor_name" not in df.columns:
        raise RuntimeError("vendor_lookup.parquet must contain: vendor_number, vendor_name")
    df["vendor_number"] = df["vendor_number"].apply(normalize_vendor_number)
    return df[["vendor_number", "vendor_name"]].drop_duplicates("vendor_number")


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    p = Path(ARTIFACT_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_model():
    if lgb is None:
        raise RuntimeError("LightGBM not available")
    p = Path(MODEL_PATH)
    if not p.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    return lgb.Booster(model_file=str(p))


@lru_cache(maxsize=1)
def get_explainer():
    return SupplierExplainer(load_features_df())


# =========================
# Feature contract + scoring
# =========================
def score_month(df_month: pd.DataFrame) -> np.ndarray:
    """
    Uses exact training feature list from artifact to avoid mismatch.
    """
    artifact = load_artifact()
    features = artifact.get("features_used") or artifact.get("feature_names")
    if not features:
        raise RuntimeError(
            f"features_used missing in artifact: {ARTIFACT_PATH}. Retrain model to generate it."
        )

    missing = [c for c in features if c not in df_month.columns]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} training features in scoring data: {missing[:25]}")

    X = df_month[features].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    model = load_model()
    return np.asarray(model.predict(X), dtype=float)


def business_score_by_rank(sorted_df: pd.DataFrame) -> pd.Series:
    """
    Converts rank into 0–100 for UI readability.
    Top row => 100, bottom => 0 (keeps ordering).
    """
    n = max(1, len(sorted_df) - 1)
    return (1.0 - (sorted_df.index / n)) * 100.0


def review_limit_from_topk(top_k: int) -> int:
    return min(REVIEW_LIMIT_DEFAULT, max(1, int(top_k)))


# =========================
# Prediction-month logic
# =========================
def available_prediction_months_from_features(df: pd.DataFrame) -> List[str]:
    """
    If features exist for months: [M1, M2, M3...]
    we can predict months: [M2, M3, ...] using collected months [M1, M2, ...]
    """
    months = sorted(df["_month"].unique().tolist())
    if len(months) <= 1:
        return []
    return months[1:]  # skip first month (no prior month to collect from)


def resolve_collected_month(predicted_month: str, df: pd.DataFrame) -> str:
    """
    predicted_month is what UI selects (month we want to forecast).
    collected_month = predicted_month - 1 month (data we use).
    """
    pred = normalize_month(predicted_month)
    collected = shift_month(pred, -1)

    # Safety: if collected isn't present, we cannot score
    existing = set(df["_month"].unique().tolist())
    if collected not in existing:
        raise RuntimeError(f"No collected data for month={collected} (needed to predict {pred})")
    return collected


# =========================
# API schemas
# =========================
class PredictRangeRequest(BaseModel):
    months: List[str]           # these are PREDICTED months from UI
    top_k: int = 200            # how many rows to return per month


# =========================
# App
# =========================
app = FastAPI(title="Supplier Risk API", version="2.0")


@app.get("/available_months")
def available_months() -> Dict[str, Any]:
    """
    Returns months that UI can request as PREDICTED months.
    """
    df = load_features_df()
    pred_months = available_prediction_months_from_features(df)
    return {"months": pred_months}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    df = load_features_df()
    pred_months = available_prediction_months_from_features(df)
    suppliers = int(df["vendor_number"].nunique())

    artifact = load_artifact()
    features_used = artifact.get("features_used") or artifact.get("feature_names") or []
    model_file = artifact.get("model_file") or artifact.get("model_path") or str(MODEL_PATH)
    training_time = int(artifact.get("training_time_seconds", 0))

    try:
        p = Path(model_file)
        model_size_mb = round(p.stat().st_size / (1024 * 1024), 4) if p.exists() else 0.0
    except Exception:
        model_size_mb = 0.0

    dataset = artifact.get("dataset", {})
    dataset_file = artifact.get("dataset_file", dataset.get("features_path", "data.xlsx"))
    dataset_size_mb = artifact.get("dataset_size_mb", dataset.get("features_size_mb", 0.0))

    return {
        "months": pred_months,
        "suppliers_count": suppliers,
        "model_file": model_file,
        "model_size_mb": float(model_size_mb),
        "dataset_file": dataset_file,
        "dataset_size_mb": float(dataset_size_mb),
        "use_metrics": features_used,
        "training_time_seconds": training_time,
        "score_definition": "Rank-based 0–100 score within predicted_month (top=100, bottom=0).",
        "top_k_default": TOP_K_DEFAULT,
        "default_top_k": TOP_K_DEFAULT,
        "review_limit_default": REVIEW_LIMIT_DEFAULT,
    }


@app.post("/predict_range")
def predict_range(req: PredictRangeRequest) -> Dict[str, Any]:
    """
    For each predicted_month P:
      collected_month C = P - 1
      score suppliers using data collected in C
      return rows annotated with:
        collected_month (C) and predicted_month (P)
    """
    df = load_features_df()
    vlookup = load_vendor_lookup()
    explainer = get_explainer()

    top_k = max(1, min(int(req.top_k), 5000))
    review_limit = review_limit_from_topk(top_k)

    months_pred = [normalize_month(m) for m in req.months]
    by_month: Dict[str, List[Dict[str, Any]]] = {}
    missing_months: List[str] = []

    for pred_month in months_pred:
        try:
            collected_month = resolve_collected_month(pred_month, df)
        except Exception:
            by_month[pred_month] = []
            missing_months.append(pred_month)
            continue

        df_c = df[df["_month"] == collected_month].copy()
        if df_c.empty:
            by_month[pred_month] = []
            missing_months.append(pred_month)
            continue

        # score for ranking
        df_c["_score_raw"] = score_month(df_c)

        df_c = df_c.merge(vlookup, on="vendor_number", how="left")
        df_c = df_c.sort_values("_score_raw", ascending=False).reset_index(drop=True)

        # human-friendly 0–100 score by rank
        df_c["score"] = business_score_by_rank(df_c)

        # decisions
        df_c["decision"] = np.where(df_c.index < review_limit, "NEED_REVIEW", "OK")

        # months
        df_c["collected_month"] = collected_month   # data month
        df_c["predicted_month"] = pred_month        # month we forecast for

        # why_flagged
        why = []
        for i, v in enumerate(df_c["vendor_number"]):
            if i < review_limit:
                why.append(explainer.explain(collected_month, v))
            else:
                why.append(
                    "• Within normal range compared to peers\n"
                    "• No action required for this supplier\n"
                    "• Continue monitoring"
                )
        df_c["why_flagged"] = why

        out_cols = [
            "vendor_name",
            "vendor_number",
            "score",
            "decision",
            "why_flagged",
            "collected_month",
            "predicted_month",
        ]

        by_month[pred_month] = df_c.head(top_k)[out_cols].to_dict(orient="records")

    return {
        "months": months_pred,
        "by_month": by_month,
        "missing_months": missing_months,
        "top_k": top_k,
        "review_limit": review_limit,
    }


@app.post("/predict")
def predict_single(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Backward-compatible endpoint for single month predictions.
    Expected payload:
      - month: 'YYYY-MM' or 'YYYY-MM-01'
      - monthly_review_limit: int (optional)
    """
    month = normalize_month(payload.get("month"))
    top_k = int(payload.get("monthly_review_limit", TOP_K_DEFAULT))
    resp = predict_range(PredictRangeRequest(months=[month], top_k=top_k))
    rows = resp["by_month"].get(month, [])
    return {"rows": rows, "results": rows}
