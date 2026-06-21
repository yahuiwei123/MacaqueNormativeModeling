"""
Shared data loading and preprocessing utilities for GPR normative modeling.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


META_COLS = {"subject_id", "age", "sex", "site", "breed", "weight (kg)", "atlas", "hemisphere"}


def lower_and_strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def ensure_columns(df: pd.DataFrame, cols: List[str], fill_value=pd.NA) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_one_merged_metric_csv(csv_path: Path) -> pd.DataFrame:
    """
    Read a merged metric CSV in wide format.
    Expected to contain at least: subject_id + age/sex/site/breed.
    Missing confounds are filled with 'unknown' / NA.
    """
    df = pd.read_csv(csv_path)
    df = lower_and_strip_columns(df)

    if "subject_id" not in df.columns:
        raise ValueError(f"Missing required column 'subject_id' in: {csv_path}")

    df = ensure_columns(df, ["age", "sex", "site", "breed"])
    df = coerce_numeric(df, ["age"])

    for c in ["sex", "site", "breed"]:
        df[c] = df[c].astype("string").fillna("unknown").str.strip()
        df.loc[df[c].isin(["", "nan", "none", "<na>"]), c] = "unknown"

    return df


def find_score_columns(
    df: pd.DataFrame,
    prefixes: Optional[tuple] = None,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify score (response variable) columns.

    If prefixes given, keep only columns starting with one of them (case-insensitive).
    Otherwise, return all non-meta columns.
    """
    exclude = set(c.lower() for c in (exclude_cols or []))

    if prefixes:
        pfx = tuple(p.lower() for p in prefixes)
        return [
            c for c in df.columns
            if c.lower() not in exclude and any(c.lower().startswith(p) for p in pfx)
        ]

    meta_lower = set(c.lower() for c in META_COLS)
    return [
        c for c in df.columns
        if c.lower() not in meta_lower and c.lower() not in exclude
    ]


def normalize_subject_id(df: pd.DataFrame, col: str = "subject_id") -> pd.DataFrame:
    """Prefix pure-digit subject IDs with 'sub-' for consistency."""
    df = df.copy()
    if col not in df.columns:
        raise ValueError(f"Missing required column '{col}' in input.")

    def _is_all_digits(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return False
        return str(x).strip().isdigit()

    s = df[col].astype(str).str.strip()
    s = s.replace({"nan": pd.NA, "None": pd.NA})
    mask = s.notna() & s.apply(_is_all_digits)
    s.loc[mask] = "sub-" + s.loc[mask].astype(str)
    df[col] = s
    return df


def infer_site_from_path(csv_path: Path) -> str:
    parent = csv_path.parent.name.strip()
    stem = csv_path.stem.strip()
    generic = {"data", "csv", "tables", "table", "phenotype", "phenotypes", "meta", "metadata"}
    if parent and parent.lower() not in generic:
        return parent
    return stem


def load_and_standardize_one(csv_path: Path) -> pd.DataFrame:
    """Load a single CSV and standardize its columns for prediction."""
    df = pd.read_csv(csv_path)
    df = lower_and_strip_columns(df)
    df = normalize_subject_id(df, "subject_id")

    if "site" not in df.columns:
        df["site"] = infer_site_from_path(csv_path)
    else:
        df["site"] = df["site"].astype("string")
        inferred = infer_site_from_path(csv_path)
        site_s = df["site"].astype(str).str.strip()
        site_s = site_s.replace({"nan": "", "None": ""})
        df.loc[site_s.eq(""), "site"] = inferred
        df["site"] = df["site"].astype("string")

    df = ensure_columns(df, ["age", "sex", "site", "breed"])
    df = coerce_numeric(df, ["age"])
    return df


def load_many(inputs: List[Path], pattern: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate multiple CSVs (files or directories)."""
    dfs = []
    for p in inputs:
        if p.is_dir():
            for csvp in sorted(p.rglob(pattern)):
                dfs.append(load_and_standardize_one(csvp))
        else:
            dfs.append(load_and_standardize_one(p))
    if not dfs:
        raise SystemExit("No CSVs loaded. Check --in_path / --pattern.")
    return pd.concat(dfs, axis=0, ignore_index=True)


def fill_missing_confounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing confound values:
    - Numeric (age): median
    - Categorical (sex, site, breed): 'unknown'
    """
    df = df.copy()
    for c in ["sex", "breed", "site"]:
        if c in df.columns:
            df[c] = df[c].astype("string").fillna("unknown").astype(str).str.strip()
        else:
            df[c] = "unknown"
    for c in ["age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            med = df[c].median(skipna=True)
            if np.isnan(med):
                med = 0.0
            df[c] = df[c].fillna(med)
    return df


def prepare_train_data(
    df: pd.DataFrame,
    score_col: str,
    conf_num: List[str],
    conf_cat: List[str],
    min_n: int = 20,
):
    """
    Prepare X, y and the filtered dataframe for training a single score column.
    Returns (X, y, filtered_df) or (None, None, None) if insufficient data.
    """
    needed = conf_num + conf_cat + [score_col, "subject_id"]
    sub = df[needed].copy()
    sub = sub.dropna(subset=conf_num + [score_col])

    if len(sub) < min_n:
        return None, None, None

    X = sub[conf_num + conf_cat]
    y = pd.to_numeric(sub[score_col], errors="coerce")
    keep = y.notna()
    sub = sub.loc[keep].reset_index(drop=True)
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].astype(float).reset_index(drop=True)

    if len(sub) < min_n:
        return None, None, None

    return X, y, sub
