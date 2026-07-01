from __future__ import annotations

from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from .constants import COMBAT_META_COLS


def get_brain_columns(df: pd.DataFrame) -> list[str]:
    meta_lower = {c.lower() for c in COMBAT_META_COLS}
    cols: list[str] = []
    for col in df.columns:
        if str(col).lower() in meta_lower:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
        elif df[col].dtype == object:
            try:
                pd.to_numeric(df[col], errors="raise")
                cols.append(col)
            except (TypeError, ValueError):
                pass
    return cols


def find_csv_files(base_dir: Path, pattern: str = "*.csv") -> list[Path]:
    files = []
    for csv_path in Path(base_dir).rglob(pattern):
        if "harmonized" in csv_path.parts:
            continue
        if "site-mri-3" in csv_path.name:
            continue
        files.append(csv_path)
    return sorted(files)


def harmonize_csv(csv_path: Path, output_dir: Path | None = None, save_model: bool = True) -> Path | None:
    from neuroHarmonize import harmonizationLearn

    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "site" not in df.columns:
        print(f"Skipping {csv_path}: missing site column")
        return None
    if "age" not in df.columns:
        print(f"Skipping {csv_path}: missing age column")
        return None

    sites = df["site"].replace("", np.nan).dropna().unique()
    if len(sites) < 2:
        print(f"Skipping {csv_path}: need at least two sites")
        return None

    brain_cols = get_brain_columns(df)
    if not brain_cols:
        print(f"Skipping {csv_path}: no numeric brain columns")
        return None

    valid_mask = df["site"].notna() & (df["site"] != "") & df["age"].notna()
    for col in brain_cols:
        valid_mask &= df[col].notna()
    df_valid = df.loc[valid_mask].copy()
    if len(df_valid) < 10:
        print(f"Skipping {csv_path}: too few valid rows ({len(df_valid)})")
        return None

    for col in brain_cols:
        if df_valid[col].dtype == object:
            df_valid[col] = pd.to_numeric(df_valid[col], errors="coerce")

    brain_data = df_valid[brain_cols].to_numpy(dtype=float)
    variances = np.var(brain_data, axis=0)
    keep = variances > 1e-12
    if not keep.all():
        removed = [brain_cols[i] for i, ok in enumerate(keep) if not ok]
        print(f"Removing {len(removed)} zero-variance features from {csv_path.name}")
        brain_cols = [brain_cols[i] for i, ok in enumerate(keep) if ok]
        brain_data = brain_data[:, keep]
    if not brain_cols:
        print(f"Skipping {csv_path}: all brain columns have zero variance")
        return None

    covars = pd.DataFrame({"SITE": df_valid["site"].values, "AGE": df_valid["age"].values})
    print(f"Harmonizing {csv_path}: {len(df_valid)} rows x {len(brain_cols)} features")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, data_adj = harmonizationLearn(brain_data, covars, smooth_terms=["AGE"])

    out_df = df_valid.copy()
    for idx, col in enumerate(brain_cols):
        out_df[col] = data_adj[:, idx]

    output_dir = Path(output_dir) if output_dir else csv_path.parent / "harmonized"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / csv_path.name
    out_df.to_csv(out_path, index=False)

    if save_model:
        import joblib

        joblib.dump(model, output_dir / f"{csv_path.stem}_combat_model.joblib")
    return out_path


def harmonize_many(
    csv_files: Iterable[Path],
    input_root: Path | None = None,
    output_root: Path | None = None,
    save_model: bool = True,
) -> list[Path]:
    outputs: list[Path] = []
    for csv_file in csv_files:
        out_dir = None
        if output_root is not None:
            if input_root is not None:
                rel_parent = Path(csv_file).parent.relative_to(input_root)
                out_dir = Path(output_root) / rel_parent
            else:
                out_dir = Path(output_root)
        out = harmonize_csv(csv_file, out_dir, save_model=save_model)
        if out:
            outputs.append(out)
    return outputs
