from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    ATLAS_LIST,
    CORTICAL_METRICS,
    DEFAULT_HEALTHY_DATA_DIR,
    DEFAULT_SAVE_DIR,
    FULL_META_COLS,
    GLOBAL_METRIC_TO_SOURCE,
    GLOBAL_VAR_MAPPING,
    HEMI_LIST,
    SPLIT_META_COLS,
    SUBCORT_GLOBAL_VARS,
)


@dataclass(frozen=True)
class DatasetSpec:
    atlas: str
    hemi: str
    metric: str
    data_path: Path
    save_dir: Path
    covariates: tuple[str, ...]
    batch_effects: tuple[str, ...]
    response_vars: tuple[str, ...]
    n_samples: int
    is_global: bool = False
    is_subcortical: bool = False

    @property
    def label(self) -> str:
        return f"{self.atlas}/{self.hemi}/{self.metric}"

    def expected_model_dir(self, full: bool = False) -> Path:
        if full:
            return self.save_dir / "full_data_model" / "model"
        return self.save_dir / "model"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["age", "sex", "breed"])
    return df.dropna(axis=1, how="all")


def filter_required_rows(
    df: pd.DataFrame,
    covariates: Iterable[str],
    batch_effects: Iterable[str],
    response_vars: Iterable[str],
) -> pd.DataFrame:
    cols = list(covariates) + list(batch_effects) + list(response_vars)
    out = df.dropna(subset=cols)
    return out[~out[cols].isin([np.inf, -np.inf]).any(axis=1)]


def read_clean_csv(path: Path) -> pd.DataFrame:
    return clean_dataframe(pd.read_csv(path))


def cortical_csv_path(
    base_dir: Path,
    atlas: str,
    hemi: str,
    metric: str,
    use_harmonized: bool,
) -> Path:
    root = base_dir / "cort" / atlas / hemi
    if use_harmonized:
        return root / "harmonized" / f"{metric}.csv"
    return root / f"{metric}.csv"


def subcort_csv_path(base_dir: Path, hemi: str, use_harmonized: bool) -> Path:
    root = base_dir / "subcort" / "aseg" / hemi
    if use_harmonized:
        harmonized = root / "harmonized" / "volume.csv"
        if harmonized.exists():
            return harmonized
    return root / "volume.csv"


def cortical_response_vars(
    df: pd.DataFrame,
    metric: str,
    meta_cols: set[str],
    include_global_covariate: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    covariates = ["age"]
    global_var = GLOBAL_VAR_MAPPING.get(metric)
    if include_global_covariate and global_var in df.columns:
        covariates.append(global_var)

    exclude = meta_cols | set(covariates) | {
        c for c in df.columns if str(c).startswith("global_")
    }
    response_vars = tuple(
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    )
    return tuple(covariates), response_vars


def subcort_response_vars(
    df: pd.DataFrame,
    hemi: str,
    meta_cols: set[str],
) -> tuple[str, ...]:
    covariates = {"age"}
    exclude = meta_cols | covariates | {
        c for c in df.columns if str(c).startswith("global_")
    }
    return tuple(
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and str(c).startswith(f"subvol_{hemi}_")
    )


def cortical_local_spec(
    base_dir: Path,
    save_base: Path,
    atlas: str,
    hemi: str,
    metric: str,
    use_harmonized: bool,
    include_global_covariate: bool,
    meta_cols: set[str] = SPLIT_META_COLS,
    min_n: int = 30,
) -> DatasetSpec | None:
    path = cortical_csv_path(base_dir, atlas, hemi, metric, use_harmonized)
    if not path.exists():
        return None
    df = read_clean_csv(path)
    covariates, response_vars = cortical_response_vars(
        df, metric, meta_cols, include_global_covariate
    )
    if not response_vars:
        return None
    filtered = filter_required_rows(df, covariates, ("sex", "breed"), response_vars)
    if len(filtered) < min_n:
        return None
    return DatasetSpec(
        atlas=atlas,
        hemi=hemi,
        metric=metric,
        data_path=path,
        save_dir=save_base / atlas / hemi / metric,
        covariates=covariates,
        batch_effects=("sex", "breed"),
        response_vars=response_vars,
        n_samples=len(filtered),
    )


def cortical_global_spec(
    base_dir: Path,
    save_base: Path,
    atlas: str,
    hemi: str,
    global_metric: str,
    use_harmonized: bool,
    min_n: int = 30,
) -> DatasetSpec | None:
    source_metric = GLOBAL_METRIC_TO_SOURCE.get(global_metric)
    if source_metric is None:
        return None
    global_var = GLOBAL_VAR_MAPPING[source_metric]
    path = cortical_csv_path(base_dir, atlas, hemi, source_metric, use_harmonized)
    if not path.exists():
        return None
    df = read_clean_csv(path)
    if global_var not in df.columns:
        return None
    response_vars = (global_var,)
    filtered = filter_required_rows(df, ("age",), ("sex", "breed"), response_vars)
    if len(filtered) < min_n:
        return None
    return DatasetSpec(
        atlas=atlas,
        hemi=hemi,
        metric=global_metric,
        data_path=path,
        save_dir=save_base / atlas / hemi / global_metric,
        covariates=("age",),
        batch_effects=("sex", "breed"),
        response_vars=response_vars,
        n_samples=len(filtered),
        is_global=True,
    )


def subcort_local_spec(
    base_dir: Path,
    save_base: Path,
    hemi: str,
    use_harmonized: bool,
    meta_cols: set[str] = SPLIT_META_COLS,
    min_n: int = 30,
) -> DatasetSpec | None:
    path = subcort_csv_path(base_dir, hemi, use_harmonized)
    if not path.exists():
        return None
    df = read_clean_csv(path)
    response_vars = subcort_response_vars(df, hemi, meta_cols)
    if not response_vars:
        return None
    filtered = filter_required_rows(df, ("age",), ("sex", "breed"), response_vars)
    if len(filtered) < min_n:
        return None
    return DatasetSpec(
        atlas="subcortical",
        hemi=hemi,
        metric="volume",
        data_path=path,
        save_dir=save_base / "subcort" / hemi / "volume",
        covariates=("age",),
        batch_effects=("sex", "breed"),
        response_vars=response_vars,
        n_samples=len(filtered),
        is_subcortical=True,
    )


def subcort_global_spec(
    base_dir: Path,
    save_base: Path,
    hemi: str,
    global_var: str,
    use_harmonized: bool,
    min_n: int = 30,
) -> DatasetSpec | None:
    path = subcort_csv_path(base_dir, hemi, use_harmonized)
    if not path.exists():
        return None
    df = read_clean_csv(path)
    if global_var not in df.columns:
        return None
    filtered = filter_required_rows(df, ("age",), ("sex", "breed"), (global_var,))
    if len(filtered) < min_n:
        return None
    return DatasetSpec(
        atlas="subcortical",
        hemi=hemi,
        metric=global_var,
        data_path=path,
        save_dir=save_base / "subcort" / hemi / global_var,
        covariates=("age",),
        batch_effects=("sex", "breed"),
        response_vars=(global_var,),
        n_samples=len(filtered),
        is_global=True,
        is_subcortical=True,
    )


def iter_cv_specs(
    base_dir: Path = DEFAULT_HEALTHY_DATA_DIR,
    save_base: Path = DEFAULT_SAVE_DIR,
    use_harmonized: bool = True,
    include_global_models: bool = True,
    include_global_covariate: bool = True,
    min_n: int = 30,
) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    base_dir = Path(base_dir)
    save_base = Path(save_base)

    for atlas in ATLAS_LIST:
        for hemi in HEMI_LIST:
            for metric in CORTICAL_METRICS:
                spec = cortical_local_spec(
                    base_dir,
                    save_base,
                    atlas,
                    hemi,
                    metric,
                    use_harmonized,
                    include_global_covariate,
                    SPLIT_META_COLS,
                    min_n,
                )
                if spec:
                    specs.append(spec)
                if include_global_models:
                    gspec = cortical_global_spec(
                        base_dir,
                        save_base,
                        atlas,
                        hemi,
                        f"global_{metric}",
                        use_harmonized,
                        min_n,
                    )
                    if gspec:
                        specs.append(gspec)

    for hemi in HEMI_LIST:
        spec = subcort_local_spec(base_dir, save_base, hemi, use_harmonized, SPLIT_META_COLS, min_n)
        if spec:
            specs.append(spec)
        if include_global_models:
            for global_var in SUBCORT_GLOBAL_VARS:
                gspec = subcort_global_spec(base_dir, save_base, hemi, global_var, use_harmonized, min_n)
                if gspec:
                    specs.append(gspec)
    return specs


def spec_from_params_row(
    row: pd.Series,
    base_dir: Path = DEFAULT_HEALTHY_DATA_DIR,
    save_base: Path = DEFAULT_SAVE_DIR,
    use_harmonized: bool = False,
    include_global_covariate: bool = False,
    min_n: int = 100,
) -> DatasetSpec | None:
    atlas = str(row["atlas"])
    hemi = str(row["hemi"])
    metric = str(row["metric"])

    if atlas == "subcortical":
        if metric == "volume":
            return subcort_local_spec(
                base_dir, save_base, hemi, use_harmonized, FULL_META_COLS, min_n
            )
        return subcort_global_spec(base_dir, save_base, hemi, metric, use_harmonized, min_n)

    if metric.startswith("global_"):
        return cortical_global_spec(base_dir, save_base, atlas, hemi, metric, use_harmonized, min_n)

    return cortical_local_spec(
        base_dir,
        save_base,
        atlas,
        hemi,
        metric,
        use_harmonized,
        include_global_covariate,
        FULL_META_COLS,
        min_n,
    )


def iter_full_specs_from_params(
    params_path: Path,
    base_dir: Path = DEFAULT_HEALTHY_DATA_DIR,
    save_base: Path = DEFAULT_SAVE_DIR,
    use_harmonized: bool = False,
    include_global_covariate: bool = False,
    min_n: int = 100,
) -> list[tuple[DatasetSpec, pd.Series]]:
    params = pd.read_csv(params_path)
    out: list[tuple[DatasetSpec, pd.Series]] = []
    for _, row in params.iterrows():
        spec = spec_from_params_row(
            row,
            base_dir=Path(base_dir),
            save_base=Path(save_base),
            use_harmonized=use_harmonized,
            include_global_covariate=include_global_covariate,
            min_n=min_n,
        )
        if spec:
            out.append((spec, row))
    return out


def model_file_names(model_dir: Path) -> set[str]:
    if not model_dir.exists():
        return set()
    return {p.parent.name for p in model_dir.glob("*/regression_model.json")}


def spec_audit_rows(specs: Iterable[DatasetSpec], full: bool = False) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        model_dir = spec.expected_model_dir(full=full)
        actual = model_file_names(model_dir)
        expected = set(spec.response_vars)
        rows.append(
            {
                "atlas": spec.atlas,
                "hemi": spec.hemi,
                "metric": spec.metric,
                "phase": "full" if full else "cv",
                "data_path": str(spec.data_path),
                "save_dir": str(spec.save_dir),
                "expected_models": len(expected),
                "actual_models": len(actual),
                "missing_models": ";".join(sorted(expected - actual)),
                "extra_models": ";".join(sorted(actual - expected)),
                "n_samples": spec.n_samples,
            }
        )
    return rows
