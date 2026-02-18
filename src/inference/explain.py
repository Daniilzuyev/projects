# src/inference/explain.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.utils.normalize import month_series_to_yyyymm, normalize_month, normalize_vendor_number

def _pick_numeric_candidates(row: pd.Series, exclude: set[str]) -> List[str]:
    cols = []
    for c, v in row.items():
        if c in exclude:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and pd.notna(v):
            cols.append(c)
    return cols


def _format_value(v: float) -> str:
    # business-friendly formatting
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    av = abs(float(v))
    if av >= 1000:
        return f"{v:,.0f}"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _humanize_feature_name(name: str) -> str:
    # keep it simple; no hard-coded mapping required
    return name.replace("_", " ").strip()


@dataclass(frozen=True)
class ExplainConfig:
    p_low: float = 0.10   # P10
    p_high: float = 0.90  # P90
    top_k: int = 3


class SupplierExplainer:
    """
    Explains a supplier-month row by comparing numeric columns to same-month peer distribution.
    Produces 3 bullet points ALWAYS.
    """

    def __init__(self, df_features: pd.DataFrame, month_col: str = "month", vendor_col: str = "vendor_number"):
        self.df = df_features.copy()

        self.month_col = month_col
        self.vendor_col = vendor_col

        # normalize keys once
        self.df[self.vendor_col] = self.df[self.vendor_col].apply(normalize_vendor_number)
        self.df["_month_yyyymm"] = month_series_to_yyyymm(self.df[self.month_col])

    def get_row(self, month: object, vendor_number: object) -> pd.Series:
        mm = normalize_month(month)
        vv = normalize_vendor_number(vendor_number)

        m = (self.df["_month_yyyymm"] == mm) & (self.df[self.vendor_col] == vv)
        if not m.any():
            raise KeyError(f"Row not found for (month={mm}, vendor_number={vv})")

        # if duplicates exist (shouldn't), take first
        return self.df.loc[m].iloc[0]

    def explain(self, month: object, vendor_number: object, cfg: ExplainConfig = ExplainConfig()) -> str:
        row = self.get_row(month, vendor_number)
        mm = row["_month_yyyymm"]

        # month peer group
        peer = self.df[self.df["_month_yyyymm"] == mm]

        exclude = {self.month_col, self.vendor_col, "_month_yyyymm"}
        numeric_cols = _pick_numeric_candidates(row, exclude=exclude)

        if not numeric_cols:
            # absolute fallback
            return "• Risk driven by combined mild deviations across multiple metrics\n" \
                   "• No single metric stands out versus peers this month\n" \
                   "• Recommend a quick review of recent transactions and compliance hygiene"

        # compute quantiles for the month, only for numeric columns that exist
        peer_num = peer[numeric_cols].apply(pd.to_numeric, errors="coerce")

        q_low = peer_num.quantile(cfg.p_low, numeric_only=True)
        q_high = peer_num.quantile(cfg.p_high, numeric_only=True)
        q_med = peer_num.quantile(0.50, numeric_only=True)

        scored: List[Tuple[float, str]] = []
        for c in numeric_cols:
            v = float(row[c]) if pd.notna(row[c]) else np.nan
            if np.isnan(v):
                continue

            lo = float(q_low.get(c, np.nan))
            hi = float(q_high.get(c, np.nan))
            med = float(q_med.get(c, np.nan))

            if np.isnan(lo) or np.isnan(hi) or np.isnan(med):
                continue

            # distance score: outside range gets strong score, inside gets softer
            if v < lo:
                dist = (lo - v) / (abs(lo - med) + 1e-9)
                scored.append((dist + 2.0, c))  # +2 to prefer out-of-range
            elif v > hi:
                dist = (v - hi) / (abs(hi - med) + 1e-9)
                scored.append((dist + 2.0, c))
            else:
                # inside range: rank by distance to median
                dist = abs(v - med) / (abs(med) + 1e-9)
                scored.append((dist, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_cols = [c for _, c in scored[: max(cfg.top_k, 3)]]

        bullets: List[str] = []
        for c in top_cols[:cfg.top_k]:
            v = float(row[c])
            lo = float(q_low[c])
            hi = float(q_high[c])

            direction = "spiked" if v > hi else ("dropped" if v < lo else "shifted")
            feat = _humanize_feature_name(c)
            bullets.append(
                f"• {feat} {direction}: {_format_value(v)} vs usual {_format_value(lo)}–{_format_value(hi)}"
            )

        # guarantee 3 bullets
        while len(bullets) < 3:
            bullets.append("• Risk driven by combined mild deviations across multiple metrics")

        return "\n".join(bullets[:3])