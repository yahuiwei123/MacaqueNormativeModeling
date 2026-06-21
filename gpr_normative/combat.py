"""
ComBat harmonization for multi-site neuroimaging data.

Removes site effects from brain region measurements while preserving
age-related biological variation (specified via smooth_terms).

Requires: pip install neuroHarmonize
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

META_COLS = {
    "participant_id", "subject_id", "session_id", "age", "sex", "site", "breed",
    "weight (kg)", "atlas", "hemisphere", "aseg_path", "voxel_volume_mm3",
}


def get_brain_columns(df: pd.DataFrame) -> list:
    """Identify numeric brain region columns (excluding metadata)."""
    meta_lower = {c.lower() for c in META_COLS}
    numeric_cols = []
    for col in df.columns:
        if col.lower() in meta_lower:
            continue
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            numeric_cols.append(col)
        elif df[col].dtype == object:
            try:
                pd.to_numeric(df[col], errors="raise")
                numeric_cols.append(col)
            except (ValueError, TypeError):
                pass
    return numeric_cols


def harmonize_csv(
    csv_path: Path,
    output_dir: Path | None = None,
) -> Path | None:
    """
    Apply ComBat harmonization to a single CSV file.

    Uses site as the batch variable and age as a smooth (preserved) term.
    Removes zero-variance features automatically.

    Returns path to harmonized CSV, or None if skipped.
    """
    from neuroHarmonize import harmonizationLearn

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if "site" not in df.columns:
        print(f"  Skipping {csv_path.name}: no 'site' column")
        return None

    sites = df["site"].replace("", np.nan).dropna().unique()
    if len(sites) < 2:
        print(f"  Skipping {csv_path.name}: only {len(sites)} site(s)")
        return None

    brain_cols = get_brain_columns(df)
    if not brain_cols:
        print(f"  Skipping {csv_path.name}: no numeric brain columns")
        return None

    # Filter valid rows (non-NaN in site, age, and all brain columns)
    valid_mask = df["site"].notna() & (df["site"] != "") & df["age"].notna()
    for col in brain_cols:
        valid_mask &= df[col].notna()
    df_valid = df[valid_mask].copy()

    n_discarded = len(df) - len(df_valid)
    if n_discarded > 0:
        print(f"  Discarded {n_discarded} rows with NaN values")

    if len(df_valid) < 10:
        print(f"  Skipping {csv_path.name}: too few valid rows ({len(df_valid)})")
        return None

    # Ensure numeric types
    for col in brain_cols:
        if df_valid[col].dtype == object:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    brain_data = df_valid[brain_cols].values.astype(float)

    # Remove zero-variance features
    variances = np.var(brain_data, axis=0)
    nonzero = variances > 1e-12
    if not nonzero.all():
        n_removed = (~nonzero).sum()
        removed_cols = [brain_cols[i] for i, ok in enumerate(nonzero) if not ok]
        print(f"  Removing {n_removed} zero-variance features: {removed_cols[:5]}...")
        brain_cols = [brain_cols[i] for i, ok in enumerate(nonzero) if ok]
        brain_data = brain_data[:, nonzero]

    if len(brain_cols) == 0:
        print(f"  Skipping {csv_path.name}: all features zero-variance")
        return None

    # ComBat: SITE as batch, AGE as smooth (preserved biological effect)
    covars = pd.DataFrame({
        "SITE": df_valid["site"].values,
        "AGE": df_valid["age"].values,
    })

    print(f"  Harmonizing: {len(df_valid)} rows × {len(brain_cols)} features × {len(sites)} sites")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, data_adj = harmonizationLearn(brain_data, covars, smooth_terms=["AGE"])

    df_output = df_valid.copy()
    for i, col in enumerate(brain_cols):
        df_output[col] = data_adj[:, i]

    if output_dir is None:
        output_dir = csv_path.parent / "harmonized"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / csv_path.name
    df_output.to_csv(out_path, index=False)

    # Save ComBat model for future harmonization of new data
    import joblib
    model_path = output_dir / (csv_path.stem + "_combat_model.joblib")
    joblib.dump(model, model_path)
    print(f"  Saved: {out_path}")
    print(f"  Model: {model_path}")
    return out_path


def harmonize_directory(
    input_dir: Path,
    output_dir: Path | None = None,
    pattern: str = "*.csv",
) -> list:
    """
    Apply ComBat to all CSVs in a directory tree.

    Returns list of output paths.
    """
    input_dir = Path(input_dir).expanduser().resolve()
    results = []
    csv_files = sorted(input_dir.rglob(pattern))
    # Exclude site-mri-3.csv and already-harmonized files
    csv_files = [
        f for f in csv_files
        if "site-mri-3" not in f.name and "harmonized" not in str(f)
    ]

    print(f"Found {len(csv_files)} CSV files to harmonize in {input_dir}")
    print("=" * 60)

    for f in csv_files:
        # Maintain relative path structure
        rel = f.relative_to(input_dir)
        out_subdir = (output_dir or input_dir / "harmonized") / rel.parent
        result = harmonize_csv(f, out_subdir)
        if result:
            results.append(result)

    print(f"\n{'=' * 60}")
    print(f"Harmonized {len(results)}/{len(csv_files)} files")
    return results


def main():
    ap = argparse.ArgumentParser(
        description="ComBat harmonization for multi-site neuroimaging data."
    )
    ap.add_argument("--csv", type=str, default="",
                    help="Single CSV file to harmonize.")
    ap.add_argument("--input_dir", type=str, default="",
                    help="Directory of CSVs to batch harmonize.")
    ap.add_argument("--output_dir", type=str, default="",
                    help="Output directory (default: <input>/harmonized).")
    args = ap.parse_args()

    if args.csv:
        out = Path(args.output_dir) if args.output_dir else None
        harmonize_csv(Path(args.csv).expanduser().resolve(), out)
    elif args.input_dir:
        out = Path(args.output_dir) if args.output_dir else None
        harmonize_directory(Path(args.input_dir), out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
