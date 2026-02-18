import pandas as pd

from src.features.build_supplier_month_table import build_supplier_month


def test_build_supplier_month_aggregates() -> None:
    df = pd.DataFrame(
        [
            {
                "date": "2022-01-05",
                "amount": 100.0,
                "vendor_number": "V1",
                "company_code": "C1",
                "category": "cat_a",
                "compliance_status": "noncompliant",
                "vendor_country": "US",
                "vendor_type": "type_a",
                "is_preferred": 1,
                "is_managed": 0,
            },
            {
                "date": "2022-01-15",
                "amount": -40.0,
                "vendor_number": "V1",
                "company_code": "C1",
                "category": "cat_b",
                "compliance_status": "noncompliant",
                "vendor_country": "US",
                "vendor_type": "type_a",
                "is_preferred": 1,
                "is_managed": 0,
            },
            {
                "date": "2022-02-01",
                "amount": 10.0,
                "vendor_number": "V2",
                "company_code": "C2",
                "category": "cat_a",
                "compliance_status": "compliant",
                "vendor_country": "DE",
                "vendor_type": "type_b",
                "is_preferred": 0,
                "is_managed": 1,
            },
        ]
    )

    out = build_supplier_month(df)

    v1_jan = out[(out["vendor_number"] == "V1") & (out["month"] == pd.Timestamp("2022-01-01"))]
    assert len(v1_jan) == 1
    row = v1_jan.iloc[0]

    assert row["tx_count"] == 2
    assert row["spend_sum"] == 60.0
    assert row["spend_abs_sum"] == 140.0
    assert row["unique_companies"] == 1
    assert row["unique_categories"] == 2
    assert row["compliance_status"] == "noncompliant"
