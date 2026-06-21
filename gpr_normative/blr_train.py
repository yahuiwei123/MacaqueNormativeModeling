"""
Train BLR (Bayesian Linear Regression) normative models with pcntoolkit.

Implements:
  1. Cross-validation for hyperparameter optimization
  2. Train/test split evaluation
  3. Full-data model training using optimal CV parameters

Requires: pip install pcntoolkit (Python 3.10-3.12)
"""

from __future__ import annotations

import argparse
import itertools
import os
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, KFold

os.environ["ARVIZ_WARNING_STAMP_FILE"] = "/dev/null"

from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData

# Suppress verbose logs
for logger_name in ["pymc", "arviz"]:
    lg = logging.getLogger(logger_name)
    lg.setLevel(logging.WARNING)
    lg.propagate = False

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None


def filter_data(data, covariates, batch_effects, response_vars):
    """Drop rows with NaN or inf in required columns."""
    cols = covariates + batch_effects + response_vars
    df = data.dropna(subset=cols)
    df = df[~df[cols].isin([np.inf, -np.inf]).any(axis=1)]
    return df


def calculate_metrics(y_true, y_pred):
    """Compute EXPV, MAE, RMSE, R², Spearman r."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"expv": np.nan, "mae": np.nan, "rmse": np.nan, "r2": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan}

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expv = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_true) > 1 else np.nan
    sr, sp = stats.spearmanr(y_true, y_pred) if len(y_true) > 2 else (np.nan, np.nan)
    return {"expv": expv, "mae": mae, "rmse": rmse, "r2": r2,
            "spearman_r": sr, "spearman_p": sp}


def create_bspline_basis(age_data, nknots, degree, adaptive_knots=True, boundary_expansion=0.05):
    """Create B-spline basis with adaptive or uniform knots."""
    if adaptive_knots:
        age_min, age_max = age_data.min(), age_data.max()
        age_range = age_max - age_min
        expanded_min = age_min - boundary_expansion * age_range
        expanded_max = age_max + boundary_expansion * age_range
        quantiles = np.linspace(0, 1, nknots + 2)
        knots = np.quantile(age_data, quantiles)
        knots[0] = expanded_min
        knots[-1] = expanded_max
        return BsplineBasisFunction(degree=degree, knots=knots)
    else:
        return BsplineBasisFunction(degree=degree, nknots=nknots)


def cross_validation_optimize(
    data, covariates, batch_effects, response_vars,
    n_folds=5, param_grid=None, random_state=42,
):
    """K-fold CV to find optimal BLR hyperparameters."""
    if param_grid is None:
        param_grid = {
            "nknots": [2, 3, 4, 5],
            "degree": [2, 3],
            "heteroskedastic": [True, False],
            "warp_name": [None],
            "adaptive_knots": [True, False],
        }

    param_names = list(param_grid.keys())
    param_combos = list(itertools.product(*param_grid.values()))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    cv_results = []

    print(f"\n{'='*60}")
    print(f"CV: {n_folds}-fold, {len(param_combos)} param combinations")
    print(f"{'='*60}")

    for param_idx, params in enumerate(param_combos):
        p = dict(zip(param_names, params))
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]

            try:
                norm_train = NormData.from_dataframe(
                    name="cv_train", dataframe=train_fold,
                    covariates=covariates, batch_effects=batch_effects,
                    response_vars=response_vars,
                )
                norm_val = NormData.from_dataframe(
                    name="cv_val", dataframe=val_fold,
                    covariates=covariates, batch_effects=batch_effects,
                    response_vars=response_vars,
                )

                basis = create_bspline_basis(
                    train_fold[covariates[0]].values,
                    nknots=p["nknots"], degree=p["degree"],
                    adaptive_knots=p["adaptive_knots"],
                )

                blr = BLR(
                    name="cv", basis_function_mean=basis,
                    fixed_effect=True,
                    heteroskedastic=p["heteroskedastic"],
                    warp_name=p["warp_name"],
                )

                nm = NormativeModel(
                    template_regression_model=blr,
                    savemodel=False, evaluate_model=False,
                    saveresults=False, saveplots=False,
                    inscaler="standardize", outscaler="standardize",
                )

                result = nm.fit_predict(norm_train, norm_val)

                if isinstance(result, tuple) and len(result) >= 2:
                    y_pred, _ = result
                    y_pred = np.asarray(y_pred.to_numpy(), dtype=np.float64)
                elif hasattr(result, "Yhat"):
                    y_pred = np.asarray(result.Yhat.to_numpy(), dtype=np.float64)
                else:
                    raise ValueError(f"Fit failed: {result}")

                y_true = np.asarray(norm_val.Y.to_numpy(), dtype=np.float64)

                roi_expvs = []
                for ri in range(y_pred.shape[1]):
                    m = calculate_metrics(y_true[:, ri], y_pred[:, ri])
                    roi_expvs.append(m["expv"])
                mean_expv = np.nanmean(roi_expvs)
                fold_scores.append(mean_expv)

            except Exception as e:
                fold_scores.append(np.nan)

        mean_score = np.nanmean(fold_scores)
        std_score = np.nanstd(fold_scores)
        cv_results.append({**p, "mean_expv": mean_score, "std_expv": std_score})
        print(f"  [{param_idx+1}/{len(param_combos)}] {p} → EXPV={mean_score:.4f} ± {std_score:.4f}")

    cv_df = pd.DataFrame(cv_results).sort_values("mean_expv", ascending=False)
    best = cv_df.iloc[0].to_dict()
    for k in ["mean_expv", "std_expv"]:
        del best[k]
    print(f"\nBest: {best}, EXPV={cv_df.iloc[0]['mean_expv']:.4f}")
    return best, cv_df


def train_blr_model(
    csv_path: Path,
    output_dir: Path,
    covariates: list = None,
    batch_effects: list = None,
    global_var: str = None,
    response_vars: list = None,
    test_size: float = 0.2,
    n_folds: int = 5,
    random_state: int = 42,
    min_n: int = 30,
):
    """
    Train a BLR normative model with CV optimization.

    Uses pcntoolkit BLR with B-spline basis + standardization.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["age", "sex", "breed"])
    df = df.dropna(axis=1, how="all")

    if covariates is None:
        covariates = ["age"]
        if global_var and global_var in df.columns:
            covariates.append(global_var)
    if batch_effects is None:
        batch_effects = ["sex", "breed"]

    if response_vars is None:
        meta = {"subject_id", "participant_id", "session_id", "age", "sex", "site", "breed",
                 "weight (kg)", "atlas", "hemisphere"}
        exclude = meta | set(covariates) | {c for c in df.columns if c.startswith("global_")}
        response_vars = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

    if not response_vars:
        raise ValueError("No response variables found.")

    df = filter_data(df, covariates, batch_effects, response_vars)

    if len(df) < min_n:
        raise ValueError(f"Not enough data: {len(df)} samples (min={min_n})")

    print(f"Training: {csv_path}")
    print(f"  Covariates: {covariates}, Batch: {batch_effects}")
    print(f"  ROIs: {len(response_vars)}, Samples: {len(df)}")

    # Train/test split
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    print(f"  Train: {len(train)}, Test: {len(test)}")

    # CV optimization
    best_params, cv_df = cross_validation_optimize(
        train, covariates, batch_effects, response_vars,
        n_folds=n_folds, random_state=random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_df.to_csv(output_dir / "cross_validation_results.csv", index=False)

    # Train with best params on train set, evaluate on test
    norm_train = NormData.from_dataframe(
        name="train", dataframe=train, covariates=covariates,
        batch_effects=batch_effects, response_vars=response_vars,
    )
    norm_test = NormData.from_dataframe(
        name="test", dataframe=test, covariates=covariates,
        batch_effects=batch_effects, response_vars=response_vars,
    )

    basis = create_bspline_basis(
        train[covariates[0]].values,
        nknots=int(best_params["nknots"]), degree=int(best_params["degree"]),
        adaptive_knots=best_params.get("adaptive_knots", True),
    )

    blr = BLR(
        name="blr", basis_function_mean=basis, fixed_effect=True,
        heteroskedastic=best_params["heteroskedastic"],
        warp_name=best_params["warp_name"],
    )

    nm = NormativeModel(
        template_regression_model=blr, savemodel=True,
        evaluate_model=True, saveresults=True, saveplots=True,
        save_dir=str(output_dir / "model"),
        inscaler="standardize", outscaler="standardize",
    )

    result = nm.fit_predict(norm_train, norm_test)

    # Per-ROI test metrics
    y_true = np.asarray(norm_test.Y.to_numpy(), dtype=np.float64)
    y_pred = np.asarray(result.Yhat.to_numpy(), dtype=np.float64)

    roi_results = []
    for i, roi in enumerate(response_vars):
        m = calculate_metrics(y_true[:, i], y_pred[:, i])
        roi_results.append({"roi": roi, **m})

    roi_df = pd.DataFrame(roi_results)
    roi_df.to_csv(output_dir / "test_metrics_by_roi.csv", index=False)

    mean_metrics = roi_df[["expv", "mae", "rmse", "r2", "spearman_r"]].mean()
    print(f"\n  Test metrics (mean of {len(response_vars)} ROIs):")
    print(f"    EXPV={mean_metrics['expv']:.4f}  MAE={mean_metrics['mae']:.4f}  "
          f"RMSE={mean_metrics['rmse']:.4f}  R²={mean_metrics['r2']:.4f}")

    return {"best_params": best_params, "test_metrics": mean_metrics.to_dict(),
            "n_rois": len(response_vars), "n_samples": len(df)}


def main():
    ap = argparse.ArgumentParser(
        description="Train BLR normative model with CV hyperparameter optimization."
    )
    ap.add_argument("--csv", type=str, required=True, help="Path to input CSV.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--covariates", type=str, default="age",
                    help="Comma-separated covariates (default: age).")
    ap.add_argument("--batch_effects", type=str, default="sex,breed",
                    help="Comma-separated batch effects (default: sex,breed).")
    ap.add_argument("--global_var", type=str, default="",
                    help="Global variable column to include as covariate.")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()

    covariates = [c.strip() for c in args.covariates.split(",") if c.strip()]
    batch_effects = [b.strip() for b in args.batch_effects.split(",") if b.strip()]
    global_var = args.global_var.strip() or None

    train_blr_model(
        csv_path=Path(args.csv).expanduser().resolve(),
        output_dir=Path(args.out_dir).expanduser().resolve(),
        covariates=covariates,
        batch_effects=batch_effects,
        global_var=global_var,
        test_size=args.test_size,
        n_folds=args.n_folds,
        random_state=args.random_state,
    )
    print(f"\n[OK] Model saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
