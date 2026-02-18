import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import src.api.app as api


def _features_df() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "vendor_number": ["V1", "V2", "V1", "V2"],
            "month": pd.to_datetime(
                ["2022-01-01", "2022-01-01", "2022-02-01", "2022-02-01"]
            ),
            "tx_count": [1, 2, 3, 4],
            "spend_sum": [10.0, 20.0, 30.0, 40.0],
        }
    )
    df["_month"] = df["month"].dt.to_period("M").dt.strftime("%Y-%m")
    return df


def _vendor_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vendor_number": ["V1", "V2"],
            "vendor_name": ["Vendor One", "Vendor Two"],
        }
    )


class _DummyExplainer:
    def explain(self, collected_month: str, vendor_number: str) -> str:
        return "ok"


def test_available_months(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_features_df", _features_df)

    client = TestClient(api.app)
    res = client.get("/available_months")
    assert res.status_code == 200
    assert res.json()["months"] == ["2022-02"]


def test_predict_range(monkeypatch) -> None:
    monkeypatch.setattr(api, "load_features_df", _features_df)
    monkeypatch.setattr(api, "load_vendor_lookup", _vendor_lookup)
    monkeypatch.setattr(api, "load_artifact", lambda: {"features_used": ["tx_count", "spend_sum"]})
    monkeypatch.setattr(api, "get_explainer", lambda: _DummyExplainer())
    monkeypatch.setattr(api, "score_month", lambda df_month: np.array([0.1, 0.9]))

    client = TestClient(api.app)
    res = client.post("/predict_range", json={"months": ["2022-02"], "top_k": 2})
    assert res.status_code == 200

    payload = res.json()
    rows = payload["by_month"]["2022-02"]
    assert rows[0]["vendor_number"] == "V2"
    assert rows[0]["collected_month"] == "2022-01"
    assert rows[0]["predicted_month"] == "2022-02"
    assert rows[0]["decision"] == "NEED_REVIEW"
