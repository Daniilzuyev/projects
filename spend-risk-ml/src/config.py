from __future__ import annotations

from pathlib import Path

from src.utils.paths import project_root

BASE_DIR = project_root()

DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_XLSX_PATH = RAW_DIR / "data.xlsx"
RAW_PARQUET_PATH = PROCESSED_DIR / "data.parquet"
SUPPLIER_MONTH_PATH = PROCESSED_DIR / "supplier_month.parquet"
LABELED_PATH = PROCESSED_DIR / "supplier_month_labeled.parquet"
FEATURES_PATH = PROCESSED_DIR / "supplier_month_features_v1.parquet"
VENDOR_LOOKUP_PATH = PROCESSED_DIR / "vendor_lookup.parquet"

MODEL_PATH = MODELS_DIR / "supplier_risk_mvp_v1.txt"
ARTIFACT_PATH = MODELS_DIR / "supplier_risk_mvp_v1_artifact.json"

TOP_K_DEFAULT = 200
REVIEW_LIMIT_DEFAULT = 50
