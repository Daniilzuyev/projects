from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import FEATURES_PATH, LABELED_PATH, MODELS_DIR

TARGET = "y_next_noncompliant"
DROP_COLS = [TARGET, "vendor_number", "month", "compliance_status"]


def main():
    labeled = pd.read_parquet(LABELED_PATH)
    feats = pd.read_parquet(FEATURES_PATH)

    # Merge (your environment currently produces _x/_y for overlapping columns)
    df = labeled.merge(
        feats,
        on=["vendor_number", "month"],
        how="inner",
        validate="one_to_one",
    )
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["month", "vendor_number"]).reset_index(drop=True)

    train_end = pd.to_datetime("2022-05-01")
    test = df[df["month"] > train_end].copy()

    print("Columns containing 'vendor_country':", [c for c in test.columns if "vendor_country" in c])
    print("Columns containing 'vendor_type':", [c for c in test.columns if "vendor_type" in c])

    y = test[TARGET].astype(int).to_numpy()

    # Load model and expected feature schema
    model = lgb.Booster(model_file=str(MODELS_DIR / "lgbm_v1.txt"))
    model_feats = model.feature_name()
    print("Model expects features:", len(model_feats))
    print("First 10 model features:", model_feats[:10])

    # Build X base
    X_raw = test.drop(columns=[c for c in DROP_COLS if c in test.columns], errors="ignore")

    # Normalize schema to match model feature names exactly:
    # For each feature name F in model, take:
    #   F_x if exists else F if exists else F_y if exists
    X = pd.DataFrame(index=X_raw.index)
    for f in model_feats:
        if f"{f}_x" in X_raw.columns:
            X[f] = X_raw[f"{f}_x"]
        elif f in X_raw.columns:
            X[f] = X_raw[f]
        elif f"{f}_y" in X_raw.columns:
            X[f] = X_raw[f"{f}_y"]
        # else: missing -> will be handled below

    # Ensure categoricals are present with the exact names model expects
    # (model_feats includes 'vendor_country'/'vendor_type' if training used them)
    for c in ["vendor_country", "vendor_type"]:
        if c in X.columns:
            X[c] = X[c].astype("category")

    print("Normalized X columns:", len(X.columns))
    print("Has vendor_country:", "vendor_country" in X.columns, "| Has vendor_type:", "vendor_type" in X.columns)

    # Add missing columns (should be rare; if missing is categorical, we can't safely fabricate)
    missing = [c for c in model_feats if c not in X.columns]
    extra = [c for c in X.columns if c not in model_feats]

    print("Missing vs model:", len(missing))
    print("Extra vs model:", len(extra))
    if missing:
        print("Missing columns (sample):", missing[:20])
        for c in missing:
            X[c] = 0

    # Reorder exactly as model expects
    X = X[model_feats]

    # Predict
    scores = model.predict(X)

    test["score"] = scores
    test["rank"] = test["score"].rank(method="first", ascending=False).astype(int)

    print("Test rows:", len(test), "Positives:", int(y.sum()))
    pos_rows = test[test[TARGET] == 1].sort_values("score", ascending=False)
    if pos_rows.empty:
        print("No positives in test.")
        return

    print("\n=== Positive event(s) position ===")
    cols = ["vendor_number", "month", "score", "rank"]
    cols = [c for c in cols if c in pos_rows.columns]
    print(pos_rows[cols].to_string(index=False))

    print("\n=== Top 20 by score (for sanity) ===")
    cols2 = ["vendor_number", "month", "score", "rank", TARGET]
    cols2 = [c for c in cols2 if c in test.columns]
    print(test.sort_values("score", ascending=False).head(20)[cols2].to_string(index=False))


if __name__ == "__main__":
    main()