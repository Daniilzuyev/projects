import pandas as pd

from src.features.pipeline import (
    FeatureConfig,
    add_compliance_history,
    add_deltas,
    add_lags,
    add_rolling,
    prepare_base,
)


def test_feature_engineering_lags_and_history() -> None:
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

    out = prepare_base(df)
    out = add_lags(out)
    out = add_deltas(out)
    out = add_rolling(out, FeatureConfig())
    out = add_compliance_history(out)

    m4 = out[out["month"] == pd.Timestamp("2022-04-01")].iloc[0]

    assert m4["is_noncompliant_t"] == 0
    assert m4["tx_count_lag1"] == 8
    assert m4["tx_count_delta_1m"] == 8 - 12
    assert m4["tx_count_mean_3m"] == (10 + 12 + 8) / 3
    assert m4["months_since_noncompliant"] == 3
    assert m4["noncompliant_last_3m"] == 1
