# Supplier Risk Early Warning (MVP)

Production-grade ML pet project for a B2B spend analytics platform: **early warning** signals when a supplier is likely to **deteriorate into noncompliance next month**.

## Problem
Procurement / compliance teams need a monthly ranked list of suppliers to review, because “true risk events” are rare and expensive to investigate.

## Target
**y_next_noncompliant = 1** if supplier transitions from:
- (compliant OR unmanaged) in month T
→ **noncompliant** in month T+1

Time-aware split is mandatory (no random split).

## Data & features (current MVP)
- Entity: `vendor_number` (supplier)
- Granularity: monthly supplier aggregates
- Window: 2022-02 → 2022-06 (model trained until 2022-05, validated on 2022-06)
- Features: 35 numeric/bool monthly features (spend & transaction stats, diversity signals, vendor flags)

Artifacts:
- `data/processed/supplier_month_labeled.parquet`
- `data/processed/supplier_month_features_v1.parquet`
- `models/supplier_risk_mvp_v1.txt`
- `models/supplier_risk_mvp_v1_artifact.json`

## Data availability
Raw and processed data files are not committed to GitHub. To run the pipeline end‑to‑end, place your raw file at `data/raw/data.xlsx` and regenerate the artifacts locally.

Runbook (data → features → train):
```bash
python -m src.data.convert_to_parquet
python -m src.data.validate_schema
python -m src.features.build_supplier_month_table
python -m src.features.build_features_v1
python -m src.models.training.train_lgbm_mvp_v1
```

## Baseline
Rule-based baseline:
> If supplier is already noncompliant this month → high risk next month

Metrics priority:
1) PR-AUC (Average Precision)
2) Recall / ranking metrics (Top-K)
(ROC-AUC is secondary)

## Model (MVP)
- Model: **LightGBM binary classifier**
- Imbalance handling: `scale_pos_weight`
- Output: supplier risk score per month, used to rank suppliers

MVP metrics (test month = 2022-06):
- PR-AUC: **0.00513**
- Best positive rank: **191**
- Recall@250: **1.0** (captures the known event within top-250)

## Alert policy (product decision)
Because risk events are extremely rare, the product uses **Top-K alerts**.
Current default:
- **top_k = 200** (minimal K that achieved Recall=1.0 on the test month)

This aligns with an “investigation budget” approach in procurement/compliance operations.

## API (FastAPI)
Endpoints:
- `GET /` → service info
- `GET /health` → healthcheck
- `GET /metadata` → model version, metrics, default top_k
- `POST /predict` → top-K supplier risk ranking for a month

Run locally:
```bash
pip install -r requirements.txt
uvicorn src.api.app:app --reload --port 8000
```

## UI (Streamlit demo)

Run API:
```bash
uvicorn src.api.app:app --reload --port 8000
streamlit run src/ui/app.py
```
