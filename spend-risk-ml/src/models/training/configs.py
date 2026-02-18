from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.config import FEATURES_PATH, LABELED_PATH, MODELS_DIR, ARTIFACT_PATH


@dataclass(frozen=True)
class TrainConfig:
    name: str
    model_path: Path
    artifact_path: Optional[Path]
    objective: str
    params: Dict[str, object]
    num_boost_round: int
    early_stopping_rounds: int
    train_end: str
    target: str = "y_next_noncompliant"
    drop_cols: List[str] = field(default_factory=lambda: ["vendor_number", "month", "compliance_status", "y_next_noncompliant"])
    drop_always: List[str] = field(
        default_factory=lambda: ["vendor_country", "vendor_type", "vendor_country_x", "vendor_country_y", "vendor_type_x", "vendor_type_y"]
    )
    cat_cols: List[str] = field(default_factory=lambda: ["vendor_country", "vendor_type"])
    use_categoricals: bool = False
    use_scale_pos_weight: bool = True
    sample_weight_pos: Optional[float] = None
    schema_version: int = 2
    features_path: Path = FEATURES_PATH
    labeled_path: Path = LABELED_PATH


def config_mvp_v1() -> TrainConfig:
    return TrainConfig(
        name="supplier_risk_mvp_v1",
        model_path=MODELS_DIR / "supplier_risk_mvp_v1.txt",
        artifact_path=ARTIFACT_PATH,
        objective="binary",
        params=dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            metric="auc",
            verbosity=-1,
            seed=42,
        ),
        num_boost_round=2000,
        early_stopping_rounds=100,
        train_end="2022-05-01",
    )


def config_lgbm_v1() -> TrainConfig:
    return TrainConfig(
        name="lgbm_v1",
        model_path=MODELS_DIR / "lgbm_v1.txt",
        artifact_path=None,
        objective="binary",
        params=dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            metric="auc",
            verbosity=-1,
            seed=42,
        ),
        num_boost_round=2000,
        early_stopping_rounds=100,
        train_end="2022-05-01",
    )


def config_lgbm_v1_1() -> TrainConfig:
    return TrainConfig(
        name="lgbm_v1_1",
        model_path=MODELS_DIR / "lgbm_v1_1.txt",
        artifact_path=None,
        objective="binary",
        params=dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=5,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            metric="auc",
            verbosity=-1,
            seed=42,
        ),
        num_boost_round=2000,
        early_stopping_rounds=100,
        train_end="2022-05-01",
        use_categoricals=True,
    )


def config_lgbm_v1_2() -> TrainConfig:
    return TrainConfig(
        name="lgbm_v1_2",
        model_path=MODELS_DIR / "lgbm_v1_2.txt",
        artifact_path=None,
        objective="binary",
        params=dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.03,
            num_leaves=31,
            min_data_in_leaf=20,
            max_depth=6,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_gain_to_split=0.1,
            lambda_l2=5.0,
            metric="auc",
            verbosity=-1,
            seed=42,
        ),
        num_boost_round=5000,
        early_stopping_rounds=200,
        train_end="2022-05-01",
        use_categoricals=False,
    )


def config_lgbm_weighted_v1() -> TrainConfig:
    return TrainConfig(
        name="lgbm_weighted_v1",
        model_path=MODELS_DIR / "lgbm_weighted_v1.txt",
        artifact_path=None,
        objective="binary",
        params=dict(
            objective="binary",
            boosting_type="gbdt",
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=10,
            max_depth=-1,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            min_gain_to_split=0.0,
            metric="auc",
            verbosity=-1,
            seed=42,
        ),
        num_boost_round=5000,
        early_stopping_rounds=200,
        train_end="2022-05-01",
        use_scale_pos_weight=False,
        sample_weight_pos=5000.0,
    )


def config_lgbm_rank_v1() -> TrainConfig:
    return TrainConfig(
        name="lgbm_rank_v1",
        model_path=MODELS_DIR / "lgbm_rank_v1.txt",
        artifact_path=None,
        objective="lambdarank",
        params=dict(
            objective="lambdarank",
            metric=["ndcg", "map"],
            learning_rate=0.05,
            num_leaves=31,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            verbosity=-1,
            seed=42,
            lambdarank_truncation_level=100,
        ),
        num_boost_round=5000,
        early_stopping_rounds=200,
        train_end="2022-05-01",
    )
