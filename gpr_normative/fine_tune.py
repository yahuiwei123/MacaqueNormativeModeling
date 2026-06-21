"""
Fine-tune pretrained GPR models on a custom dataset.

This is useful when you have a small new dataset and want to adapt
existing normative models rather than training from scratch.

Approach:
  1. Load a pretrained model (.joblib)
  2. Optionally freeze certain kernel hyperparameters
  3. Re-fit on new data
  4. Save the fine-tuned model

GPR Hyperparameter Guide (for the ConstantKernel * Matern + WhiteKernel):
  - constant_value: Overall signal variance. Higher = more flexible curve.
  - length_scale: How quickly the function varies with age. Smaller = wigglier.
  - noise_level: White noise variance. Higher = more tolerance for noisy data.
  - nu: Matern smoothness (set at train time, not refit). 0.5=rough, 1.5=smooth, 2.5=very smooth, inf=RBF.

Typical fine-tuning strategy:
  - Small new dataset (<50 subjects): freeze length_scale, refit constant_value + noise_level
  - Medium dataset (50-200): refit all hyperparameters with more restarts
  - Large dataset (>200): consider training from scratch
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, Kernel

from .data_utils import (
    load_one_merged_metric_csv,
    find_score_columns,
    prepare_train_data,
)


def rebuild_kernel(
    base_kernel: Kernel,
    nu: float | None = None,
    freeze_length_scale: bool = False,
) -> Kernel:
    """
    Rebuild a kernel from a pretrained model's kernel, optionally freezing parts.

    The pretrained kernel has structure: ConstantKernel * Matern + WhiteKernel
    """
    prod = base_kernel.k1  # ConstantKernel * Matern
    white = base_kernel.k2  # WhiteKernel

    ck = prod.k1  # ConstantKernel
    matern = prod.k2  # Matern

    new_nu = nu if nu is not None else matern.nu

    # Get current hyperparameter values as starting points
    cv = ck.get_params()["constant_value"]
    ls = matern.get_params()["length_scale"]
    nl = white.get_params()["noise_level"]

    if freeze_length_scale:
        # Fix length_scale to its trained value
        new_matern = Matern(length_scale=ls, length_scale_bounds="fixed", nu=new_nu)
    else:
        new_matern = Matern(
            length_scale=ls,
            length_scale_bounds=(ls * 0.1, ls * 10.0),
            nu=new_nu,
        )

    new_ck = ConstantKernel(cv, constant_value_bounds=(cv * 0.1, cv * 10.0))
    new_white = WhiteKernel(nl, noise_level_bounds=(nl * 0.01, nl * 100.0))

    return new_ck * new_matern + new_white


def fine_tune_one_score(
    df: pd.DataFrame,
    score_col: str,
    model_path: Path,
    out_dir: Path,
    conf_num: list,
    conf_cat: list,
    freeze_length_scale: bool,
    n_restarts: int,
    random_state: int,
    min_n: int = 10,
) -> dict | None:
    """Fine-tune a single pretrained model on new data."""
    X, y, sub = prepare_train_data(df, score_col, conf_num, conf_cat, min_n)
    if X is None:
        return None

    pipe = load(model_path)
    preproc = pipe.named_steps["preproc"]
    old_gpr = pipe.named_steps["gpr"]

    # Rebuild kernel with current values as starting points
    # kernel_ structure: Sum(Product(ConstantKernel, Matern), WhiteKernel)
    matern_kernel = old_gpr.kernel_.k1.k2  # Product.k2 = Matern
    new_kernel = rebuild_kernel(
        old_gpr.kernel_,
        nu=matern_kernel.nu,
        freeze_length_scale=freeze_length_scale,
    )

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.pipeline import Pipeline

    new_gpr = GaussianProcessRegressor(
        kernel=new_kernel,
        alpha=old_gpr.alpha,
        normalize_y=old_gpr.normalize_y,
        n_restarts_optimizer=n_restarts,
        random_state=random_state,
    )

    new_pipe = Pipeline([("preproc", preproc), ("gpr", new_gpr)])
    new_pipe.fit(X, y)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    dump(new_pipe, model_dir / f"{score_col}.joblib")

    mu, std = new_pipe.predict(X, return_std=True)
    std = np.maximum(std, 1e-8)
    from .train import compute_metrics
    metrics = compute_metrics(y.values, mu)

    # Kernel params after fine-tuning
    final_kernel = new_pipe.named_steps["gpr"].kernel_
    params = {
        "constant_value": final_kernel.k1.k1.get_params()["constant_value"],
        "length_scale": final_kernel.k1.k2.get_params()["length_scale"],
        "noise_level": final_kernel.k2.get_params()["noise_level"],
    }

    return {"score_col": score_col, "n_samples": len(y), **metrics, **params}


def main():
    ap = argparse.ArgumentParser(
        description="Fine-tune pretrained GPR models on a custom dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fine-tune all models with default settings
  python -m gpr_normative.fine_tune --csv my_data.csv --model_dir pretrained/models --out_dir finetuned

  # Freeze length_scale (good for small datasets)
  python -m gpr_normative.fine_tune --csv my_data.csv --model_dir pretrained/models --out_dir finetuned --freeze_length_scale

  # More optimizer restarts for better fit
  python -m gpr_normative.fine_tune --csv my_data.csv --model_dir pretrained/models --out_dir finetuned --n_restarts 10
        """,
    )
    ap.add_argument("--csv", type=str, required=True, help="Custom dataset CSV.")
    ap.add_argument("--model_dir", type=str, required=True, help="Directory of pretrained *.joblib models.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for fine-tuned models.")

    ap.add_argument("--score_prefixes", type=str, default="",
                    help="Comma-separated score column prefixes to fine-tune.")
    ap.add_argument("--exclude_cols", type=str, default="",
                    help="Comma-separated columns to exclude.")

    ap.add_argument("--conf_num", type=str, default="age", help="Numeric confounds.")
    ap.add_argument("--conf_cat", type=str, default="sex,site,breed", help="Categorical confounds.")

    ap.add_argument("--freeze_length_scale", action="store_true",
                    help="Keep length_scale fixed at pretrained value (recommended for n<50).")
    ap.add_argument("--n_restarts", type=int, default=5,
                    help="Optimizer restarts (more = better fit, slower).")
    ap.add_argument("--min_n", type=int, default=10,
                    help="Minimum subjects for fine-tuning a score.")
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--random_state", type=int, default=0)

    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    model_files = sorted(model_dir.glob("*.joblib"))
    if not model_files:
        raise SystemExit(f"No .joblib files found in: {model_dir}")

    df = load_one_merged_metric_csv(csv_path)
    df.columns = [c.lower() for c in df.columns]

    conf_num = [s.strip().lower() for s in args.conf_num.split(",") if s.strip()]
    conf_cat = [s.strip().lower() for s in args.conf_cat.split(",") if s.strip()]

    prefixes = tuple(s.strip().lower() for s in args.score_prefixes.split(",") if s.strip()) or None
    exclude_cols = [s.strip().lower() for s in args.exclude_cols.split(",") if s.strip()]
    score_cols = find_score_columns(df, prefixes, exclude_cols=exclude_cols)

    # Only fine-tune scores that have a matching pretrained model
    available_scores = {mp.stem for mp in model_files}
    scores_to_tune = [sc for sc in score_cols if sc in available_scores]

    if not scores_to_tune:
        raise SystemExit(f"No matching score columns between CSV and models in {model_dir}")

    if len(scores_to_tune) < len(score_cols):
        skipped = set(score_cols) - set(scores_to_tune)
        print(f"Note: {len(skipped)} score column(s) have no pretrained model and will be skipped.")

    print(f"Fine-tuning {len(scores_to_tune)} score columns...")
    print(f"  freeze_length_scale: {args.freeze_length_scale}")
    print(f"  n_restarts: {args.n_restarts}")

    from joblib import Parallel, delayed
    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(fine_tune_one_score)(
            df, sc, model_dir / f"{sc}.joblib", out_dir,
            conf_num, conf_cat,
            args.freeze_length_scale, args.n_restarts, args.random_state, args.min_n,
        )
        for sc in scores_to_tune
    )

    valid = [r for r in results if r is not None]
    results_df = pd.DataFrame(valid)
    results_df.to_csv(out_dir / "fine_tune_results.csv", index=False)

    print(f"\n[OK] Fine-tuned {len(valid)}/{len(scores_to_tune)} models")
    print(f"[OK] Results saved to: {out_dir}/")
    if len(valid):
        print(f"  Mean EXPV: {results_df['expv'].mean():.4f}")
        print(f"  Mean R²:   {results_df['r2'].mean():.4f}")


if __name__ == "__main__":
    main()
