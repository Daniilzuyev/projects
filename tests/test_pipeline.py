import pandas as pd

from src.features.pipeline import FeatureConfig, build_feature_table


def test_build_feature_table_adds_expected_columns() -> None:
    df = pd.DataFrame(
        {
            "vendor_number": ["V1"] * 4,
            "month": pd.to_datetime(["2022-01-01", "2022-02-01", "2022-03-01", "2022-04-01"]),
            "tx_count": [10, 12, 8, 9],
            "spend_sum": [100.0, 120.0, 80.0, 90.0],
            "spend_abs_sum": [100.0, 120.0, 80.0, 90.0],
            "compliance_status": ["noncompliant", "compliant", "compliant", "compliant"],
        }
    )

    out = build_feature_table(df, FeatureConfig(rolling_window=3))

    expected = [
        "tx_count_lag1",
        "spend_sum_lag1",
        "tx_count_delta_1m",
        "spend_sum_delta_1m",
        "tx_count_mean_3m",
        "tx_count_std_3m",
        "tx_count_cv_3m",
        "months_since_noncompliant",
        "noncompliant_last_3m",
    ]
    for col in expected:
        assert col in out.columns
