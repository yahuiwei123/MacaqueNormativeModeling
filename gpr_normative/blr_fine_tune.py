"""
Fine-tune pretrained BLR normative models on a custom dataset.

Loads a trained pcntoolkit NormativeModel and re-fits on new data.

Strategy:
  - Load existing model config (basis, standardization)
  - Optionally modify B-spline hyperparameters (nknots, degree)
  - Re-fit BLR on new data
  - Save fine-tuned model

Requires: pip install pcntoolkit (Python 3.10-3.12)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData


def fine_tune_blr(
    model_dir: Path,
    csv_path: Path,
    output_dir: Path,
    nknots: int | None = None,
    degree: int | None = None,
    heteroskedastic: bool | None = None,
    n_restarts: int = 5,
    random_state: int = 42,
):
    """
    Fine-tune a trained BLR model on new data.

    Args:
        model_dir: Path to trained BLR model (contains model/normative_model.json).
        csv_path: Path to new dataset CSV.
        output_dir: Where to save fine-tuned model.
        nknots: Override B-spline knots. None = use original.
        degree: Override B-spline degree. None = use original.
        heteroskedastic: Override heteroskedastic. None = use original.
        n_restarts: Number of optimizer restarts for BLR fit.
        random_state: Random seed.
    """
    # Load new data
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["age", "sex", "breed"])
    df = df.dropna(axis=1, how="all")

    # Load original model to get config
    nm = NormativeModel.load(str(model_dir))

    covariates = nm.covariates
    batch_effects = list(nm.unique_batch_effects.keys()) if hasattr(nm, "unique_batch_effects") and nm.unique_batch_effects else ["sex", "breed"]
    response_vars = nm.response_vars

    # Filter response_vars to those present in new data
    response_vars = [r for r in response_vars if r in df.columns]
    if not response_vars:
        raise ValueError("No matching response variables in new data")

    # Filter data
    cols = covariates + batch_effects + response_vars
    df = df.dropna(subset=cols)
    df = df[~df[cols].isin([np.inf, -np.inf]).any(axis=1)]

    print(f"Fine-tuning on {len(df)} samples, {len(response_vars)} ROIs")
    print(f"  Covariates: {covariates}, Batch: {batch_effects}")

    # Extract original B-spline config
    orig_bf = nm.template_regression_model.basis_function_mean
    use_nknots = nknots if nknots is not None else getattr(orig_bf, "nknots", 5)
    use_degree = degree if degree is not None else getattr(orig_bf, "degree", 3)
    use_hetero = heteroskedastic if heteroskedastic is not None else getattr(
        nm.template_regression_model, "heteroskedastic", False
    )
    use_warp = getattr(nm.template_regression_model, "warp_name", None)

    # Create B-spline basis adapted to new age distribution
    age_data = df[covariates[0]].values
    age_min, age_max = age_data.min(), age_data.max()
    age_range = age_max - age_min
    quantiles = np.linspace(0, 1, use_nknots + 2)
    knots = np.quantile(age_data, quantiles)
    knots[0] = age_min - 0.05 * age_range
    knots[-1] = age_max + 0.05 * age_range

    print(f"  B-spline: nknots={use_nknots}, degree={use_degree}, "
          f"heteroskedastic={use_hetero}")
    print(f"  Knots: {[round(k, 2) for k in knots]}")

    basis = BsplineBasisFunction(degree=use_degree, knots=knots)

    blr = BLR(
        name="blr_finetuned", basis_function_mean=basis,
        fixed_effect=True, heteroskedastic=use_hetero,
        warp_name=use_warp,
    )

    # Create NormData
    norm_data = NormData.from_dataframe(
        name="finetune_data", dataframe=df,
        covariates=covariates, batch_effects=batch_effects,
        response_vars=response_vars,
    )

    # Fit and save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        nm_new = NormativeModel(
            template_regression_model=blr, savemodel=True,
            evaluate_model=True, saveresults=True, saveplots=True,
            save_dir=str(output_dir / "model"),
            inscaler="standardize", outscaler="standardize",
        )

        result = nm_new.fit_predict(norm_data, norm_data)

    # Compute metrics on new data
    y_true = np.asarray(norm_data.Y.to_numpy(), dtype=np.float64)
    y_pred = np.asarray(result.Yhat.to_numpy(), dtype=np.float64)

    roi_results = []
    for i, roi in enumerate(response_vars):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        expv = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        mae = np.mean(np.abs(y_true[:, i] - y_pred[:, i]))
        sr, sp = stats.spearmanr(y_true[:, i], y_pred[:, i])
        roi_results.append({"roi": roi, "expv": expv, "mae": mae,
                            "spearman_r": sr, "spearman_p": sp})

    roi_df = pd.DataFrame(roi_results)
    roi_df.to_csv(output_dir / "fine_tune_metrics.csv", index=False)

    mean_expv = roi_df["expv"].mean()
    mean_mae = roi_df["mae"].mean()
    print(f"\n  Mean EXPV: {mean_expv:.4f}")
    print(f"  Mean MAE:  {mean_mae:.4f}")
    print(f"  Top 3 ROIs by EXPV:")
    for _, r in roi_df.nlargest(3, "expv").iterrows():
        print(f"    {r['roi']}: EXPV={r['expv']:.4f}")
    print(f"\n[OK] Fine-tuned model saved to: {output_dir}")

    return roi_df


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune a pretrained BLR normative model on custom data."
    )
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to pretrained BLR model directory.")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to new dataset CSV.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output directory for fine-tuned model.")
    ap.add_argument("--nknots", type=int, default=None,
                    help="Override B-spline knots (default: keep original).")
    ap.add_argument("--degree", type=int, default=None,
                    help="Override B-spline degree (default: keep original).")
    ap.add_argument("--heteroskedastic", type=str, default=None,
                    choices=["true", "false"],
                    help="Override heteroskedastic modeling (default: keep original).")

    args = ap.parse_args()

    hetero = None
    if args.heteroskedastic is not None:
        hetero = args.heteroskedastic.lower() == "true"

    fine_tune_blr(
        model_dir=Path(args.model_dir).expanduser().resolve(),
        csv_path=Path(args.csv).expanduser().resolve(),
        output_dir=Path(args.out_dir).expanduser().resolve(),
        nknots=args.nknots,
        degree=args.degree,
        heteroskedastic=hetero,
    )


if __name__ == "__main__":
    main()
