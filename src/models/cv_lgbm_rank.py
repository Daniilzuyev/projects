from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import lightgbm as lgb

from src.config import FEATURES_PATH, LABELED_PATH

TARGET = "y_next_noncompliant"
DROP_COLS = [TARGET, "vendor_number", "month", "compliance_status"]


def load_join() -> pd.DataFrame:
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
    return df


def make_Xy(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    X = X.drop(columns=[c for c in X.columns if c.endswith("_feat")], errors="ignore")
    # numeric only for stability
    X = X.select_dtypes(include=[np.number, "bool"])
    y = df[TARGET].astype(int).to_numpy()
    return X, y


def rank_of_positives(y_true: np.ndarray, scores: np.ndarray) -> list[int]:
    order = np.argsort(-scores)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)  # 1-based rank
    pos_idx = np.where(y_true == 1)[0]
    return sorted([int(ranks[i]) for i in pos_idx])


def eval_fold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    ap = average_precision_score(y_true, scores)
    pos_ranks = rank_of_positives(y_true, scores)
    return {
        "ap": float(ap),
        "pos_count": int(y_true.sum()),
        "pos_ranks": pos_ranks,
        "best_pos_rank": min(pos_ranks) if pos_ranks else None,
        "median_pos_rank": int(np.median(pos_ranks)) if pos_ranks else None,
    }


def main():
    df = load_join()
    X, y = make_Xy(df)

    months = sorted(df["month"].unique())
    print("Months:", [str(m) for m in months])
    print("Total rows:", len(df), "Total positives:", int(y.sum()), "Rate:", float(y.mean()))
    print("Features:", X.shape[1])

    # Define time folds manually: train up to month i, validate on month i+1
    # With months 2022-02..2022-06: folds will be:
    # train<=2022-03 -> val=2022-04
    # train<=2022-04 -> val=2022-05
    # train<=2022-05 -> val=2022-06
    folds = []
    for i in range(1, len(months) - 0):
        if i + 1 >= len(months):
            break
        train_end = months[i]
        val_month = months[i + 1]
        folds.append((train_end, val_month))

    print("\nFolds (train_end -> val_month):")
    for tr, va in folds:
        print(tr, "->", va)

    params = dict(
        objective="binary",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        metric="auc",  # internal
        verbosity=-1,
        seed=42,
    )

    results = []
    for train_end, val_month in folds:
        train_idx = df["month"] <= train_end
        val_idx = df["month"] == val_month

        X_tr, y_tr = X.loc[train_idx], y[train_idx]
        X_va, y_va = X.loc[val_idx], y[val_idx]

        pos = y_tr.sum()
        neg = len(y_tr) - pos
        spw = (neg / pos) if pos > 0 else 1.0

        fold_params = dict(params)
        fold_params["scale_pos_weight"] = float(spw)

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dvalid = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        model = lgb.train(
            params=fold_params,
            train_set=dtrain,
            num_boost_round=3000,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
        )

        scores = model.predict(X_va, num_iteration=model.best_iteration)
        m = eval_fold(y_va, scores)
        m.update(
            {
                "train_end": str(train_end),
                "val_month": str(val_month),
                "train_rows": int(train_idx.sum()),
                "val_rows": int(val_idx.sum()),
                "train_pos": int(pos),
                "val_pos": int(y_va.sum()),
                "best_iter": int(model.best_iteration),
                "spw": float(spw),
            }
        )
        results.append(m)

        print(
            f"\nFold train<= {train_end.date()} val= {val_month.date()} | "
            f"val_pos={m['pos_count']} | AP={m['ap']:.6f} | "
            f"best_pos_rank={m['best_pos_rank']} | it={m['best_iter']} | spw={spw:.1f}"
        )

    # Summary
    aps = [r["ap"] for r in results]
    best_ranks = [r["best_pos_rank"] for r in results if r["best_pos_rank"] is not None]
    print("\n=== CV Summary ===")
    print("Mean AP:", float(np.mean(aps)) if aps else None)
    print("Best ranks across folds:", best_ranks)


if __name__ == "__main__":
    main()
