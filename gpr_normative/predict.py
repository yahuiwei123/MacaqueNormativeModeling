"""
Predict deviations (mu, std, resid, z-score) using trained GPR models.

Loads trained .joblib models and applies them to new subject data.
Outputs per-score prediction CSVs and a merged wide table.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from joblib import load

from .data_utils import load_and_standardize_one, load_many, fill_missing_confounds


def list_model_files(model_dir: Path) -> List[Path]:
    return sorted(model_dir.glob("*.joblib"))


def predict_one_metric(
    df: pd.DataFrame,
    model_path: Path,
    out_dir: Path,
    conf_num: List[str],
    conf_cat: List[str],
    id_cols: List[str],
    min_n: int = 1,
) -> Dict[str, object]:
    score_col = model_path.stem

    if score_col not in df.columns:
        return {"score_col": score_col, "status": "skipped_missing_score_column",
                "n_pred": 0, "model": str(model_path)}

    needed = conf_num + conf_cat + [score_col]
    sub = df[id_cols + needed].copy()
    sub = sub.dropna(subset=conf_num + conf_cat + [score_col])

    if len(sub) < min_n:
        return {"score_col": score_col, "status": "skipped_too_few_rows",
                "n_pred": len(sub), "model": str(model_path)}

    X = sub[conf_num + conf_cat]
    y = pd.to_numeric(sub[score_col], errors="coerce").astype(float)
    ok = y.notna()
    sub = sub.loc[ok].copy()
    y = y.loc[ok]
    X = X.loc[ok]

    if len(sub) < min_n:
        return {"score_col": score_col, "status": "skipped_too_few_after_coerce",
                "n_pred": len(sub), "model": str(model_path)}

    pipe = load(model_path)
    mu, std = pipe.predict(X, return_std=True)
    std = np.maximum(std, 1e-8)
    resid = y.values - mu
    z = resid / std

    out = sub[id_cols + conf_num + conf_cat + [score_col]].copy()
    out[f"{score_col}__mu"] = mu
    out[f"{score_col}__std"] = std
    out[f"{score_col}__resid"] = resid
    out[f"{score_col}__z"] = z

    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{score_col}__predictions.csv"
    out.to_csv(out_path, index=False)

    return {"score_col": score_col, "status": "ok", "n_pred": len(out),
            "model": str(model_path), "out_csv": str(out_path)}


def main():
    ap = argparse.ArgumentParser(
        description="Predict deviation scores using trained GPR models."
    )
    ap.add_argument("--in_path", type=str, required=True,
                    help="Input CSV file or directory containing CSVs.")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Directory containing *.joblib trained models.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--pattern", type=str, default="*.csv",
                    help="Glob pattern when in_path is a directory.")
    ap.add_argument("--fill_missing", action="store_true",
                    help="Fill missing confounds instead of dropping rows.")
    ap.add_argument("--min_n", type=int, default=1,
                    help="Minimum rows to produce predictions for a metric.")
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_many([in_path], args.pattern)

    if args.fill_missing:
        df = fill_missing_confounds(df)

    conf_num = ["age"]
    conf_cat = ["sex", "site", "breed"]

    id_cols = ["subject_id"]
    for c in ["group"]:
        if c in df.columns and c not in id_cols:
            id_cols.append(c)

    model_files = list_model_files(model_dir)
    if not model_files:
        raise SystemExit(f"No model files (*.joblib) found in: {model_dir}")

    summaries = []
    for mp in model_files:
        summaries.append(predict_one_metric(
            df=df, model_path=mp, out_dir=out_dir,
            conf_num=conf_num, conf_cat=conf_cat,
            id_cols=id_cols, min_n=args.min_n,
        ))

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out_dir / "predict_summary.csv", index=False)

    # Merge all predictions into a single wide table
    ok_files = [s.get("out_csv") for s in summaries if s.get("status") == "ok" and s.get("out_csv")]
    if ok_files:
        key_cols = id_cols
        merged_wide = None
        for f in ok_files:
            t = pd.read_csv(f)
            keep = key_cols + [c for c in t.columns
                               if c not in key_cols
                               and c not in conf_num + conf_cat
                               and c.endswith(("__mu", "__std", "__resid", "__z"))]
            t2 = t[keep].copy()
            merged_wide = t2 if merged_wide is None else merged_wide.merge(t2, on=key_cols, how="outer")
        if merged_wide is not None:
            merged_wide.to_csv(out_dir / "predictions_merged_wide.csv", index=False)

    n_ok = sum(1 for s in summaries if s.get("status") == "ok")
    print(f"[OK] Predictions written to: {out_dir}")
    print(f"[OK] {n_ok}/{len(summaries)} score columns predicted successfully.")


if __name__ == "__main__":
    main()
