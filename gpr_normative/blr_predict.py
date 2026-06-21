"""
Predict using trained pcntoolkit BLR normative models.

Requires pcntoolkit >= 1.1.0. Install with: pip install pcntoolkit

BLR model directory structure (from blr/save_dir):
  {atlas}/{hemi}/{metric}/
  ├── model/
  │   ├── normative_model.json          # shared config, scalers, covariate info
  │   ├── {ROI_NAME}/
  │   │   └── regression_model.json     # fitted BLR per ROI
  │   └── ...
  ├── full_data_model/                  # (optional) full-data trained version
  │   └── model/...
  └── cross_validation_results.csv

Usage:
  python -m gpr_normative.blr_predict \
      --model_dir path/to/M129/L/thickness \
      --csv new_subjects.csv \
      --out_dir results/blr_predictions
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def predict_blr_models(
    model_dir: Path,
    df: pd.DataFrame,
    roi_names: List[str] | None = None,
) -> dict:
    """
    Load BLR models with pcntoolkit and compute predictions.

    Args:
        model_dir: Path to {atlas}/{hemi}/{metric} directory (contains model/ subdir).
        df: Subject data with columns matching the model's covariates + batch_effects.
        roi_names: Subset of ROIs. None = all available.

    Returns:
        dict with keys:
          'predictions': DataFrame (subject_id, roi, roi__mu, roi__z, roi__resid)
          'summary': DataFrame (per-ROI status)
    """
    from pcntoolkit.normative_model import NormativeModel
    from pcntoolkit.dataio.norm_data import NormData

    # Load the normative model wrapper
    nm = NormativeModel.load(str(model_dir))

    response_vars = nm.response_vars
    if roi_names is not None:
        response_vars = [r for r in roi_names if r in response_vars]

    if not response_vars:
        raise ValueError(f"No matching ROIs found in {model_dir}")

    covariates = nm.covariates

    # Determine batch effects from model config
    batch_effects = []
    if hasattr(nm, 'unique_batch_effects') and nm.unique_batch_effects:
        batch_effects = list(nm.unique_batch_effects.keys())

    # Ensure required columns exist
    required_cols = covariates + batch_effects + response_vars
    df_filtered = df.copy()
    for c in required_cols:
        if c not in df_filtered.columns:
            if c in covariates:
                raise ValueError(f"Missing covariate column: {c}")
            # Missing ROI column: add NaN
            df_filtered[c] = np.nan

    # Drop rows with NaN in covariates/batch effects
    df_filtered = df_filtered.dropna(subset=covariates + batch_effects)

    if len(df_filtered) == 0:
        raise ValueError("No valid rows after filtering NaN covariates")

    # Create NormData
    norm_data = NormData.from_dataframe(
        name="predict_data",
        dataframe=df_filtered,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )

    # Predict
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = nm.predict(norm_data)

    # Extract predictions
    y_pred = np.asarray(result.Yhat.to_numpy(), dtype=np.float64)
    y_true = np.asarray(norm_data.Y.to_numpy(), dtype=np.float64)

    # Build output
    results = []
    id_cols = ["subject_id"]
    for c in ["group", "session_id"]:
        if c in df_filtered.columns:
            id_cols.append(c)
    id_df = df_filtered[id_cols].reset_index(drop=True)

    for i, roi in enumerate(response_vars):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        resid = yt - yp
        std_est = np.nanstd(resid)
        if std_est < 1e-8:
            std_est = 1.0
        z = resid / std_est

        row = id_df.copy()
        row["roi"] = roi
        row["observed"] = yt
        row["predicted_mu"] = yp
        row["residual"] = resid
        row["z_score"] = z
        results.append(row)

    full_df = pd.concat(results, axis=0, ignore_index=True)
    summary = pd.DataFrame([
        {"roi": roi, "status": "ok", "n_pred": len(id_df)}
        for roi in response_vars
    ])

    return {"predictions": full_df, "summary": summary}


def main():
    ap = argparse.ArgumentParser(
        description="Predict using trained pcntoolkit BLR normative models."
    )
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to BLR model directory (e.g. M129/L/thickness).")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to input CSV with subject data.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory.")
    ap.add_argument("--rois", type=str, default="",
                    help="Comma-separated ROI names to predict. Empty = all.")

    args = ap.parse_args()

    try:
        import pcntoolkit  # noqa
    except ImportError:
        raise SystemExit(
            "pcntoolkit is required for BLR prediction.\n"
            "Install with: pip install pcntoolkit\n"
            "Note: pcntoolkit requires Python 3.10-3.12."
        )

    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df.columns = [c.strip().lower() for c in df.columns]

    roi_names = [r.strip() for r in args.rois.split(",") if r.strip()] or None

    result = predict_blr_models(model_dir, df, roi_names)

    metric_name = model_dir.name
    out_path = out_dir / f"{metric_name}__blr_predictions.csv"
    result["predictions"].to_csv(out_path, index=False)
    result["summary"].to_csv(out_dir / f"{metric_name}__blr_summary.csv", index=False)

    n_rois = result["summary"]["roi"].nunique()
    print(f"[OK] Predicted {n_rois} ROIs for {len(result['predictions']) // n_rois} subjects")
    print(f"[OK] Output: {out_path}")


if __name__ == "__main__":
    main()
