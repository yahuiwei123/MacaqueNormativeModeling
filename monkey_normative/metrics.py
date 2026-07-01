from __future__ import annotations

import numpy as np
from scipy import stats


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics used by the legacy BLR scripts."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) < 2:
        return {
            "expv": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        }

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expv = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_true) > 1 else np.nan
    spearman_r, spearman_p = (
        stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (np.nan, np.nan)
    )

    return {
        "expv": expv,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }
