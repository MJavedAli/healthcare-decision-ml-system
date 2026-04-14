from __future__ import annotations

from typing import Dict
import pandas as pd


def compare_feature_means(reference: pd.DataFrame, current: pd.DataFrame, threshold: float = 0.2) -> Dict[str, dict]:
    """Very small drift check for interviews and demos.

    For each column, compare mean shift. This is not a full drift framework,
    but it shows the production idea: baseline vs live data.
    """
    report: Dict[str, dict] = {}
    common_cols = [c for c in reference.columns if c in current.columns]
    for col in common_cols:
        ref_mean = float(reference[col].mean())
        cur_mean = float(current[col].mean())
        denom = abs(ref_mean) if abs(ref_mean) > 1e-9 else 1.0
        rel_change = abs(cur_mean - ref_mean) / denom
        report[col] = {
            "reference_mean": ref_mean,
            "current_mean": cur_mean,
            "relative_change": float(rel_change),
            "drifted": bool(rel_change > threshold),
        }
    return report
