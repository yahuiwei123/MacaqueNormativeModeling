"""
Train Gaussian Process Regression normative models.

Each score (ROI) column gets its own GPR model saved as a .joblib file.
Models use ConstantKernel * Matern + WhiteKernel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from joblib import dump, Parallel, delayed

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import train_test_split

from .data_utils import (
    load_one_merged_metric_csv,
    find_score_columns,
    prepare_train_data,
)


def build_gpr_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    nu: float = 2.5,
    n_restarts_optimizer: int = 1,
    random_state: int = 0,
) -> Pipeline:
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_cols),
            ("cat", ohe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )

    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=nu)
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e2))
    )

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )

    return Pipeline([("preproc", preproc), ("gpr", gpr)])


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute regression metrics."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    expv = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2 if len(y_true) > 1 else np.nan
    return {"expv": expv, "mae": mae, "rmse": rmse, "r2": r2, "n": len(y_true)}


def train_one_score(
    df: pd.DataFrame,
    score_col: str,
    conf_num: List[str],
    conf_cat: List[str],
    out_dir: Path,
    nu: float,
    n_restarts_optimizer: int,
    random_state: int,
    min_n: int = 20,
    test_size: float = 0.0,
) -> Optional[dict]:
    """
    Train a GPR model for one score column. Optionally evaluate on a test split.
    """
    X, y, sub = prepare_train_data(df, score_col, conf_num, conf_cat, min_n)
    if X is None:
        return None

    test_metrics = None
    if 0.0 < test_size < 1.0 and len(y) >= 10:
        train_idx, test_idx = train_test_split(
            np.arange(len(y)), test_size=test_size, random_state=random_state
        )
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    pipe = build_gpr_pipeline(
        numeric_cols=conf_num,
        categorical_cols=conf_cat,
        nu=nu,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    pipe.fit(X_train, y_train)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(pipe, model_dir / f"{score_col}.joblib")

    # Predict on all data
    mu_all, std_all = pipe.predict(X, return_std=True)
    std_all = np.maximum(std_all, 1e-8)
    z_all = (y.values - mu_all) / std_all
    resid_all = y.values - mu_all

    out_pred = sub.copy()
    out_pred[f"{score_col}__mu"] = mu_all
    out_pred[f"{score_col}__std"] = std_all
    out_pred[f"{score_col}__z"] = z_all
    out_pred[f"{score_col}__resid"] = resid_all

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(pred_dir / f"{score_col}__train_predictions.csv", index=False)

    # Test evaluation
    if X_test is not None and len(X_test) > 0:
        mu_test = mu_all[test_idx]
        test_metrics = compute_metrics(y_test.values, mu_test)
        test_metrics["mean_z"] = float(np.mean(z_all[test_idx]))
        test_metrics["std_z"] = float(np.std(z_all[test_idx]))

    result = {"score_col": score_col, "n_train": len(y_train)}
    if test_metrics:
        result["test_metrics"] = test_metrics
    return result


def main():
    ap = argparse.ArgumentParser(
        description="Train GPR normative models from a merged metric CSV (wide format)."
    )
    ap.add_argument("--csv", type=str, required=True, help="Path to merged metric CSV.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for models and predictions.")

    ap.add_argument("--score_prefixes", type=str, default="",
                    help="Comma-separated score column prefixes. If empty, use all non-meta columns.")
    ap.add_argument("--exclude_cols", type=str, default="",
                    help="Comma-separated columns to exclude from scores.")

    ap.add_argument("--conf_num", type=str, default="age", help="Comma-separated numeric confounds.")
    ap.add_argument("--conf_cat", type=str, default="sex,site,breed", help="Comma-separated categorical confounds.")

    ap.add_argument("--nu", type=float, default=2.5, help="Matern kernel smoothness parameter.")
    ap.add_argument("--n_restarts", type=int, default=1, help="Optimizer restarts for GPR.")
    ap.add_argument("--min_n", type=int, default=20, help="Minimum subjects for training a score.")
    ap.add_argument("--n_jobs", type=int, default=4, help="Parallel jobs.")
    ap.add_argument("--random_state", type=int, default=0)
    ap.add_argument("--test_size", type=float, default=0.0,
                    help="Fraction of data to hold out for evaluation (0 = no holdout).")

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_one_merged_metric_csv(csv_path)
    conf_num = [s.strip().lower() for s in args.conf_num.split(",") if s.strip()]
    conf_cat = [s.strip().lower() for s in args.conf_cat.split(",") if s.strip()]

    # Normalize column names for consistent access
    df.columns = [c.lower() for c in df.columns]

    for c in conf_num + conf_cat + ["subject_id"]:
        if c not in df.columns:
            raise SystemExit(f"Missing required column '{c}' in CSV: {csv_path}")

    prefixes = tuple(s.strip().lower() for s in args.score_prefixes.split(",") if s.strip()) or None
    exclude_cols = [s.strip().lower() for s in args.exclude_cols.split(",") if s.strip()]

    score_cols = find_score_columns(df, prefixes, exclude_cols=exclude_cols)
    if not score_cols:
        raise SystemExit("No score columns selected. Check --score_prefixes / --exclude_cols.")

    df.to_csv(out_dir / "train_table.csv", index=False)

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(train_one_score)(
            df, sc, conf_num, conf_cat, out_dir,
            args.nu, args.n_restarts, args.random_state, args.min_n, args.test_size,
        )
        for sc in score_cols
    )

    trained = [r for r in results if r is not None]
    (out_dir / "trained_score_cols.txt").write_text(
        "\n".join(r["score_col"] for r in trained) + "\n"
    )

    # Save evaluation summary if test_size was used
    if args.test_size > 0:
        eval_rows = []
        for r in trained:
            if "test_metrics" in r:
                row = {"score_col": r["score_col"], "n_train": r["n_train"]}
                row.update(r["test_metrics"])
                eval_rows.append(row)
        if eval_rows:
            pd.DataFrame(eval_rows).to_csv(out_dir / "test_evaluation.csv", index=False)

    print(f"[OK] input: {csv_path}")
    print(f"[OK] score cols found: {len(score_cols)}")
    print(f"[OK] models trained: {len(trained)} (min_n={args.min_n})")
    print(f"[OK] outputs under: {out_dir}")


if __name__ == "__main__":
    main()
