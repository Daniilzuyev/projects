from __future__ import annotations

import requests
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Supplier Risk Early Warning", layout="wide")
st.title("Supplier Risk Early Warning — MVP")


# -------------------------
# HTTP helpers
# -------------------------
def safe_get_json(url: str, timeout: int = 10):
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"GET {url} failed: {r.status_code}: {r.text}"
    except requests.RequestException as e:
        return None, f"GET {url} failed: {e}"


def safe_post_json(url: str, payload: dict, timeout: int = 30):
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json(), None
        return None, f"POST {url} failed: {r.status_code}: {r.text}"
    except requests.RequestException as e:
        return None, f"POST {url} failed: {e}"


def month_key(m: str) -> str:
    # canonical month key for UI+API join: 'YYYY-MM'
    if not m:
        return ""
    s = str(m).strip()
    return s[:7]


def month_label(m: str) -> str:
    # nice label for tabs
    return month_key(m)


# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("API URL", value="http://127.0.0.1:8000")

metadata, meta_err = safe_get_json(f"{api_url}/metadata", timeout=10)
available, months_err = safe_get_json(f"{api_url}/available_months", timeout=10)

# API returns months like: ["2022-02","2022-03",...]
months_api = []
if available and isinstance(available.get("months"), list):
    months_api = available["months"]

# UI uses month pickers that previously had 'YYYY-MM-01' – keep that UX but stable mapping
months_list = [f"{m}-01" for m in months_api] if months_api else []
if not months_list:
    months_list = ["2022-02-01", "2022-03-01", "2022-04-01", "2022-05-01", "2022-06-01"]

default_top_k = int((metadata or {}).get("top_k_default", (metadata or {}).get("default_top_k", 200)))

with st.sidebar.expander("Model info", expanded=True):
    if metadata:
        dataset_file = metadata.get("dataset_file", "data.xlsx")
        dataset_size_mb = float(metadata.get("dataset_size_mb", 0.0))
        training_seconds = int(metadata.get("training_time_seconds", 0))
        suppliers_count = int(metadata.get("suppliers_count", 0))
        use_metrics = metadata.get("use_metrics") or []

        st.write(f"**Model file:** {dataset_file} ({dataset_size_mb:.2f} Mb)")
        st.write("**Model type:** LightGBM (tree-based risk model)")
        st.write("**Use metrics:**")
        if use_metrics:
            st.write(", ".join([f"`{c}`" for c in use_metrics[:30]]) + (" ..." if len(use_metrics) > 30 else ""))
        else:
            st.write("N/A")

        # show time in a simple way
        h = training_seconds // 3600
        m = (training_seconds % 3600) // 60
        s = training_seconds % 60
        st.write(f"**Trained time:** {h:02d}:{m:02d}:{s:02d}")
        st.write(f"**Suppliers count:** {suppliers_count}")
    else:
        st.warning(meta_err or "No metadata available yet.")

if months_err:
    st.sidebar.warning(months_err)


# -------------------------
# Page description
# -------------------------
st.markdown(
    """
## What this page about:

**Purpose:** This model is trained to **rank suppliers by the risk of compliance deterioration next month**,
so a compliance/procurement team can **review the most risky suppliers first**.

**Score model:** **LightGBM (Gradient Boosted Decision Trees)**

**How it works (simple):**
- Builds many small decision trees step-by-step
- Each next tree focuses on mistakes made by previous trees
- Final score = combined signal from all trees  
➡️ Output is a **ranking score** used to order suppliers from higher risk to lower risk (within the selected month).
"""
)

st.markdown("### Were selected next fields from `data.xlsx` file:")
st.markdown(
    """
**Supplier-month activity & spend**
- `tx_count` — number of transactions in the month  
- `spend_sum` — total spend in the month  
- `spend_mean` — average transaction spend  
- `spend_std` — spend volatility (standard deviation)  
- `spend_max` / `spend_min` — extreme values  
- `unique_companies` — number of companies buying from this supplier  
- `unique_categories` — number of categories  

**Supplier attributes**
- `vendor_country` — supplier country (categorical)  
- `vendor_type` — supplier type (categorical)  
- `is_preferred` — preferred supplier flag  
- `is_managed` — managed supplier flag  
"""
)
st.markdown("""
### How score is calculated (business-friendly)

We rank all suppliers **within the selected collected_month** by the model's risk signal (higher = riskier).
Then we convert the rank into a **0–100 score** so it’s easy to read.

**Formula:**

`score = (1 - rank_index / (total_suppliers - 1)) * 100`

Where:
- `rank_index` = position in the sorted list (**0 for the #1 supplier**)
- `total_suppliers` = number of suppliers in that month

**Example for the #1 supplier:**
- `rank_index = 0`  
- `score = (1 - 0/(total_suppliers - 1)) * 100 = 100`

So the top supplier always gets **score = 100**, and lower-ranked suppliers get proportionally smaller scores.
""")



# -------------------------
# Scoring window controls
# -------------------------
st.sidebar.subheader("Scoring window")
mode = st.sidebar.radio("Mode", options=["Single month", "Multiple months"], index=1)

if mode == "Single month":
    month = st.sidebar.selectbox("Month", options=months_list, index=len(months_list) - 1)
    months = [month]
else:
    default_sel = months_list[-3:] if len(months_list) >= 3 else months_list
    months = st.sidebar.multiselect("Months", options=months_list, default=default_sel)

review_limit = st.sidebar.slider("Monthly review limit", min_value=10, max_value=500, value=default_top_k, step=10)


# -------------------------
# Rendering
# -------------------------
def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    preferred = ["vendor_name", "score", "decision", "why_flagged", "compliance_status", "vendor_number"]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    return out[cols]


def render_month_block(month_str: str, rows: list[dict]):
    raw = pd.DataFrame(rows)
    if raw.empty:
        st.info("No rows returned for this month.")
        return

    # normalize & sort
    if "score" in raw.columns:
        raw = raw.sort_values("score", ascending=False).reset_index(drop=True)

    # enforce limit (Monthly review limit)
    raw = raw.head(int(review_limit)).copy()

    # if API didn't provide decision/why_flagged, keep columns but do not break UI
    if "decision" not in raw.columns:
        raw["decision"] = "REVIEW"
    if "why_flagged" not in raw.columns:
        raw["why_flagged"] = ""

    # UI-friendly month label
    st.subheader(f"Supplier Watchlist — {month_label(month_str)}")
    st.dataframe(order_columns(raw), use_container_width=True, hide_index=True)


fetch = st.button("Fetch predictions", type="primary")
if not fetch:
    st.info("Click **Fetch predictions** to call the API and render results.")
    st.stop()

if not months:
    st.warning("Select at least one month.")
    st.stop()


# -------------------------
# Single month
# -------------------------
if len(months) == 1:
    payload = {"month": months[0], "monthly_review_limit": int(review_limit)}
    data, err = safe_post_json(f"{api_url}/predict", payload, timeout=30)
    if err:
        st.error(err)
        st.stop()

    st.success("Model: supplier_risk_mvp_v1 is generated!")

    # API compatibility: accept either {rows:[...]} or {results:[...]}
    rows = data.get("rows")
    if rows is None:
        rows = data.get("results", [])

    render_month_block(months[0], rows)


# -------------------------
# Multiple months
# -------------------------
else:
    payload = {"months": months, "top_k": int(review_limit)}
    data, err = safe_post_json(f"{api_url}/predict_range", payload, timeout=60)
    if err:
        st.error(err)
        st.stop()

    st.success("Model: supplier_risk_mvp_v1 is generated!")

    # API compatibility: prefer by_month, fallback to results_by_month
    by_month = data.get("by_month")
    if by_month is None:
        by_month = data.get("results_by_month", {})

    # Tabs labels use 'YYYY-MM'
    tab_labels = [month_label(m) for m in months]
    tabs = st.tabs(tab_labels)

    for i, m in enumerate(months):
        key = month_key(m)  # 'YYYY-MM'
        with tabs[i]:
            month_rows = (by_month or {}).get(key, [])
            render_month_block(m, month_rows)
