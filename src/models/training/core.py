from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.models.training.configs import TrainConfig


def recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    order = np.argsort(-scores)
    topk = y_true[order][:k]
    pos = y_true.sum()
    if pos == 0:
        return np.nan
    return float(topk.sum() / pos)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(y_true))
    order = np.argsort(-scores)
    topk = y_true[order][:k]
    return float(topk.mean()) if k > 0 else np.nan


def best_pos_rank(y_true: np.ndarray, scores: np.ndarray) -> int | None:
    if y_true.sum() == 0:
        return None
    order = np.argsort(-scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_idx = np.where(y_true == 1)[0]
    return int(min(ranks[pos_idx]))


def hash_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_join(features_path: Path, labeled_path: Path, target: str) -> pd.DataFrame:
    labeled = pd.read_parquet(labeled_path)
    feats = pd.read_parquet(features_path)

    if target not in labeled.columns:
        raise ValueError(f"Target `{target}` not found in labeled data: {labeled_path}")

    df = labeled.merge(
        feats,
        on=["vendor_number", "month"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_feat"),
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["month", "vendor_number"]).reset_index(drop=True)
    return df


def select_features(
    df: pd.DataFrame,
    target: str,
    drop_cols: List[str],
    drop_always: List[str],
    use_categoricals: bool,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if c.endswith("_feat")], errors="ignore")
    X = X.drop(columns=[c for c in drop_always if c in X.columns], errors="ignore")

    cat_features = []
    if use_categoricals:
        for c in cat_cols:
            if c in X.columns:
                X[c] = X[c].astype("category")
                cat_features.append(c)

    allowed = list(X.select_dtypes(include=[np.number, "bool"]).columns) + list(cat_features)
    X = X[allowed]

    y = df[target].astype(int).to_numpy()
    return X, y, cat_features


def time_split(df: pd.DataFrame, train_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.to_datetime(train_end)
    train = df[df["month"] <= cutoff].copy()
    test = df[df["month"] > cutoff].copy()
    return train, test


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    pos = y_train.sum()
    neg = len(y_train) - pos
    return (neg / pos) if pos > 0 else 1.0


def train_binary(config: TrainConfig) -> Dict[str, object]:
    t0 = time.time()
    df = load_join(config.features_path, config.labeled_path, config.target)
    X, y, cat_features = select_features(
        df,
        config.target,
        config.drop_cols,
        config.drop_always,
        config.use_categoricals,
        config.cat_cols,
    )

    train_df, test_df = time_split(df, config.train_end)
    X_train, y_train, _ = select_features(
        train_df,
        config.target,
        config.drop_cols,
        config.drop_always,
        config.use_categoricals,
        config.cat_cols,
    )
    X_test, y_test, _ = select_features(
        test_df,
        config.target,
        config.drop_cols,
        config.drop_always,
        config.use_categoricals,
        config.cat_cols,
    )

    params = dict(config.params)
    if config.use_scale_pos_weight and "scale_pos_weight" not in params:
        params["scale_pos_weight"] = float(compute_scale_pos_weight(y_train))

    weights = None
    if config.sample_weight_pos is not None:
        weights = np.where(y_train == 1, float(config.sample_weight_pos), 1.0)

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weights,
        categorical_feature=cat_features if config.use_categoricals else None,
    )
    dvalid = lgb.Dataset(
        X_test,
        label=y_test,
        reference=dtrain,
        categorical_feature=cat_features if config.use_categoricals else None,
    )

    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=config.num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True)],
    )

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    ap = float(average_precision_score(y_test, y_pred))
    best_rank = best_pos_rank(y_test, y_pred)

    metrics = {
        "pr_auc": ap,
        "best_positive_rank": best_rank,
        "recall_at_10": recall_at_k(y_test, y_pred, 10),
        "recall_at_25": recall_at_k(y_test, y_pred, 25),
        "recall_at_50": recall_at_k(y_test, y_pred, 50),
        "recall_at_100": recall_at_k(y_test, y_pred, 100),
        "recall_at_250": recall_at_k(y_test, y_pred, 250),
        "precision_at_10": precision_at_k(y_test, y_pred, 10),
        "precision_at_25": precision_at_k(y_test, y_pred, 25),
        "precision_at_50": precision_at_k(y_test, y_pred, 50),
        "precision_at_100": precision_at_k(y_test, y_pred, 100),
        "precision_at_250": precision_at_k(y_test, y_pred, 250),
    }

    save_model_and_artifact(
        config,
        model,
        metrics,
        training_time=int(time.time() - t0),
        features_used=list(X.columns),
        test_months=sorted(test_df["month"].astype(str).unique().tolist()),
    )

    return metrics


def train_rank(config: TrainConfig) -> Dict[str, object]:
    t0 = time.time()
    df = load_join(config.features_path, config.labeled_path, config.target)
    train_df, test_df = time_split(df, config.train_end)

    X_train, y_train, _ = select_features(
        train_df,
        config.target,
        config.drop_cols,
        config.drop_always,
        config.use_categoricals,
        config.cat_cols,
    )
    X_test, y_test, _ = select_features(
        test_df,
        config.target,
        config.drop_cols,
        config.drop_always,
        config.use_categoricals,
        config.cat_cols,
    )

    train_groups = train_df.groupby("month").size().to_list()
    test_groups = test_df.groupby("month").size().to_list()

    dtrain = lgb.Dataset(X_train, label=y_train, group=train_groups)
    dvalid = lgb.Dataset(X_test, label=y_test, group=test_groups, reference=dtrain)

    model = lgb.train(
        params=config.params,
        train_set=dtrain,
        num_boost_round=config.num_boost_round,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=config.early_stopping_rounds, verbose=True)],
    )

    scores = model.predict(X_test, num_iteration=model.best_iteration)
    best_rank = best_pos_rank(y_test, scores)

    metrics = {
        "best_positive_rank": best_rank,
        "train_groups": train_groups,
        "test_groups": test_groups,
    }

    save_model_and_artifact(
        config,
        model,
        metrics,
        training_time=int(time.time() - t0),
        features_used=list(X_train.columns),
        test_months=sorted(test_df["month"].astype(str).unique().tolist()),
    )

    return metrics


def save_model_and_artifact(
    config: TrainConfig,
    model: lgb.Booster,
    metrics: Dict[str, object],
    training_time: int,
    features_used: List[str],
    test_months: List[str],
) -> None:
    config.model_path.parent.mkdir(exist_ok=True)
    model.save_model(str(config.model_path))

    if config.artifact_path is None:
        return

    labeled_md5 = hash_file(config.labeled_path)
    features_md5 = hash_file(config.features_path)
    artifact = {
        "schema_version": config.schema_version,
        "model_file": str(config.model_path),
        "training_time_seconds": int(training_time),
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "features_used": features_used,
        "target": config.target,
        "objective": config.objective,
        "params": config.params,
        "data_window": {
            "train_end": config.train_end,
            "test_months": test_months,
        },
        "dataset": {
            "features_path": str(config.features_path),
            "labels_path": str(config.labeled_path),
            "features_md5": features_md5,
            "labels_md5": labeled_md5,
            "features_size_mb": round(config.features_path.stat().st_size / (1024 * 1024), 4) if config.features_path.exists() else 0.0,
            "labels_size_mb": round(config.labeled_path.stat().st_size / (1024 * 1024), 4) if config.labeled_path.exists() else 0.0,
        },
        "metrics": metrics,
        "train_config": {
            "name": config.name,
            "use_categoricals": config.use_categoricals,
            "cat_cols": config.cat_cols,
            "drop_cols": config.drop_cols,
            "drop_always": config.drop_always,
            "use_scale_pos_weight": config.use_scale_pos_weight,
            "sample_weight_pos": config.sample_weight_pos,
        },
    }

    config.artifact_path.write_text(json.dumps(artifact, indent=2))
