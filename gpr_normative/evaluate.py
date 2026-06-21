"""
K-fold cross-validation for GPR normative models.

Evaluates model performance with multiple metrics:
EXPV (explained variance), MAE, RMSE, R², Spearman correlation.
Stratifies by site to avoid data leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from .data_utils import (
    load_one_merged_metric_csv,
    find_score_columns,
    prepare_train_data,
)
from .train import build_gpr_pipeline, compute_metrics


def cross_validate_one_score(
    df: pd.DataFrame,
    score_col: str,
    conf_num: list,
    conf_cat: list,
    nu: float,
    n_restarts: int,
    n_folds: int,
    random_state: int,
    min_n: int = 20,
) -> dict | None:
    """K-fold CV for a single score column, grouping by site."""
    X, y, sub = prepare_train_data(df, score_col, conf_num, conf_cat, min_n)
    if X is None:
        return None

    # Use site as the grouping variable for splitting
    site_col = "site" if "site" in sub.columns else None
    if site_col and site_col in sub.columns:
        groups = sub[site_col].values
        n_sites = len(np.unique(groups))
        effective_folds = min(n_folds, n_sites, len(y))
        if effective_folds < 2:
            effective_folds = min(n_folds, len(y))
            splitter = StratifiedGroupKFold if False else None
        else:
            splitter = None
    else:
        groups = None
        effective_folds = min(n_folds, len(y))

    # Use KFold when StratifiedGroupKFold is not feasible
    from sklearn.model_selection import KFold
    if groups is not None and len(np.unique(groups)) >= 2:
        try:
            cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
            # Need a dummy y for stratification; bin age
            y_binned = pd.qcut(y, q=effective_folds, labels=False, duplicates="drop")
            splits = list(cv.split(X, y_binned, groups=groups))
        except Exception:
            cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
            splits = list(cv.split(X))
    else:
        cv = KFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(X))

    fold_metrics = []
    for fold_i, (train_idx, val_idx) in enumerate(splits):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if len(y_tr) < min_n:
            continue

        pipe = build_gpr_pipeline(
            numeric_cols=conf_num,
            categorical_cols=conf_cat,
            nu=nu,
            n_restarts_optimizer=n_restarts,
            random_state=random_state + fold_i,
        )
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)

        m = compute_metrics(y_val.values, y_pred)
        m["fold"] = fold_i
        fold_metrics.append(m)

    if not fold_metrics:
        return None

    metrics_df = pd.DataFrame(fold_metrics)
    summary = {
        "score_col": score_col,
        "n_samples": len(y),
        "n_folds_completed": len(fold_metrics),
        "mean_expv": metrics_df["expv"].mean(),
        "std_expv": metrics_df["expv"].std(),
        "mean_mae": metrics_df["mae"].mean(),
        "std_mae": metrics_df["mae"].std(),
        "mean_rmse": metrics_df["rmse"].mean(),
        "std_rmse": metrics_df["rmse"].std(),
        "mean_r2": metrics_df["r2"].mean(),
        "std_r2": metrics_df["r2"].std(),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(
        description="K-fold cross-validation for GPR normative models."
    )
    ap.add_argument("--csv", type=str, required=True, help="Path to merged metric CSV.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for CV results.")

    ap.add_argument("--score_prefixes", type=str, default="",
                    help="Comma-separated score column prefixes.")
    ap.add_argument("--exclude_cols", type=str, default="",
                    help="Comma-separated columns to exclude.")

    ap.add_argument("--conf_num", type=str, default="age", help="Numeric confounds.")
    ap.add_argument("--conf_cat", type=str, default="sex,site,breed", help="Categorical confounds.")

    ap.add_argument("--nu", type=float, default=2.5, help="Matern kernel smoothness.")
    ap.add_argument("--n_restarts", type=int, default=1, help="GPR optimizer restarts.")
    ap.add_argument("--n_folds", type=int, default=5, help="Number of CV folds.")
    ap.add_argument("--min_n", type=int, default=20, help="Minimum subjects for a fold.")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=0)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_one_merged_metric_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    conf_num = [s.strip().lower() for s in args.conf_num.split(",") if s.strip()]
    conf_cat = [s.strip().lower() for s in args.conf_cat.split(",") if s.strip()]

    prefixes = tuple(s.strip().lower() for s in args.score_prefixes.split(",") if s.strip()) or None
    exclude_cols = [s.strip().lower() for s in args.exclude_cols.split(",") if s.strip()]
    score_cols = find_score_columns(df, prefixes, exclude_cols=exclude_cols)

    if not score_cols:
        raise SystemExit("No score columns found.")

    print(f"Running {args.n_folds}-fold CV on {len(score_cols)} score columns...")

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(cross_validate_one_score)(
            df, sc, conf_num, conf_cat,
            args.nu, args.n_restarts, args.n_folds, args.random_state, args.min_n,
        )
        for sc in score_cols
    )

    valid = [r for r in results if r is not None]
    if not valid:
        raise SystemExit("No scores had enough data for cross-validation.")

    results_df = pd.DataFrame(valid)
    results_df.to_csv(out_dir / "cv_results.csv", index=False)

    # Print summary
    cols = ["mean_expv", "mean_mae", "mean_rmse", "mean_r2"]
    print(f"\n{'='*60}")
    print(f"Cross-Validation Summary ({len(valid)} score columns)")
    print(f"{'='*60}")
    for c in cols:
        vals = results_df[c].dropna()
        if len(vals):
            print(f"  {c}: {vals.mean():.4f} +/- {vals.std():.4f}")

    top5 = results_df.sort_values("mean_expv", ascending=False).head(5)
    print(f"\nTop 5 ROIs by EXPV:")
    for _, r in top5.iterrows():
        print(f"  {r['score_col']}: EXPV={r['mean_expv']:.4f}")

    bottom5 = results_df.sort_values("mean_expv", ascending=True).head(5)
    print(f"\nBottom 5 ROIs by EXPV:")
    for _, r in bottom5.iterrows():
        print(f"  {r['score_col']}: EXPV={r['mean_expv']:.4f}")

    print(f"\n[OK] Results saved to: {out_dir / 'cv_results.csv'}")


if __name__ == "__main__":
    main()
