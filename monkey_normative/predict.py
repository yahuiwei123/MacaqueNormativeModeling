from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd

from .blr import create_bspline_basis
from .data import filter_required_rows, read_clean_csv
from .metrics import calculate_metrics


os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-pcntoolkit")


def resolve_model_dir(
    model_dir: Path | None = None,
    save_base: Path | None = None,
    atlas: str | None = None,
    hemi: str | None = None,
    metric: str | None = None,
    full: bool = True,
) -> Path:
    if model_dir is not None:
        path = Path(model_dir).expanduser().resolve()
        if (path / "model" / "normative_model.json").exists():
            return path
        if full and (path / "full_data_model" / "model" / "normative_model.json").exists():
            return path / "full_data_model"
        raise FileNotFoundError(f"No pcntoolkit model found under {path}")

    if not (save_base and atlas and hemi and metric):
        raise ValueError("Provide either model_dir or save_base+atlas+hemi+metric")
    root = Path(save_base)
    if atlas == "subcortical":
        path = root / "subcort" / hemi / metric
    else:
        path = root / atlas / hemi / metric
    if full:
        path = path / "full_data_model"
    if not (path / "model" / "normative_model.json").exists():
        raise FileNotFoundError(f"No pcntoolkit model found under {path}")
    return path


def _load_normative_model(model_dir: Path, tmp_save_dir: Path | None = None):
    from pcntoolkit.normative_model import NormativeModel

    model = NormativeModel.load(str(model_dir))
    model.evaluate_model = False
    model.saveresults = False
    model.saveplots = False
    if tmp_save_dir is not None:
        tmp_save_dir.mkdir(parents=True, exist_ok=True)
        model.save_dir = str(tmp_save_dir)
    return model


def _batch_effects(model) -> list[str]:
    if hasattr(model, "unique_batch_effects") and model.unique_batch_effects:
        return list(model.unique_batch_effects.keys())
    if hasattr(model, "batch_effects_maps") and model.batch_effects_maps:
        return list(model.batch_effects_maps.keys())
    return ["sex", "breed"]


def _case_insensitive_rename(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    lower_map = {str(c).lower(): c for c in out.columns}
    renames = {}
    for col in required:
        if col not in out.columns and col.lower() in lower_map:
            renames[lower_map[col.lower()]] = col
    if renames:
        out = out.rename(columns=renames)
    return out


def _dataarray_to_frame(data, name: str) -> pd.DataFrame:
    if name not in data:
        return pd.DataFrame()
    obj = data[name]
    frame = obj.to_pandas()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()
    return frame.reset_index(drop=True)


def predict_deviations(
    model_dir: Path,
    csv_path: Path,
    out_dir: Path,
    roi_names: list[str] | None = None,
    allow_missing_responses: bool = False,
) -> dict[str, Path]:
    from pcntoolkit.dataio.norm_data import NormData

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = _load_normative_model(Path(model_dir), out_dir / "_pcntoolkit_tmp")

    df = pd.read_csv(csv_path)
    response_vars = list(model.response_vars)
    if roi_names:
        wanted = set(roi_names)
        response_vars = [roi for roi in response_vars if roi in wanted]
    if not response_vars:
        raise ValueError("No matching response variables in model")

    covariates = list(model.covariates)
    batch_effects = _batch_effects(model)
    df = _case_insensitive_rename(df, covariates + batch_effects + response_vars + ["subject_id"])

    missing_covars = [c for c in covariates + batch_effects if c not in df.columns]
    if missing_covars:
        raise ValueError(f"Missing covariate/batch columns: {missing_covars}")

    missing_responses = [r for r in response_vars if r not in df.columns]
    if missing_responses and not allow_missing_responses:
        raise ValueError(
            "Missing response columns needed for deviation z-scores: "
            + ", ".join(missing_responses[:10])
        )
    for roi in missing_responses:
        df[roi] = np.nan

    if "subject_id" not in df.columns:
        df["subject_id"] = [f"row_{i}" for i in range(len(df))]

    required = covariates + batch_effects
    df = df.dropna(subset=required).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows after dropping missing covariates/batch effects")

    norm_data = NormData.from_dataframe(
        name="prediction",
        dataframe=df,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
        subject_ids="subject_id",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.predict(norm_data)

    yhat = _dataarray_to_frame(result, "Yhat")
    zscores = _dataarray_to_frame(result, "Z")
    logp = _dataarray_to_frame(result, "logp")
    observed = norm_data.Y.to_pandas().reset_index(drop=True)
    if isinstance(observed, pd.Series):
        observed = observed.to_frame()

    rows = []
    id_cols = [c for c in ["subject_id", "session_id", "group"] if c in df.columns]
    id_df = df[id_cols].reset_index(drop=True)
    id_df.insert(0, "row_id", np.arange(len(id_df)))
    pivot_index = ["row_id"] + id_cols
    for roi in response_vars:
        row = id_df.copy()
        row["roi"] = roi
        row["observed"] = observed[roi].to_numpy() if roi in observed else np.nan
        row["predicted_mu"] = yhat[roi].to_numpy() if roi in yhat else np.nan
        row["z_score"] = zscores[roi].to_numpy() if roi in zscores else np.nan
        row["residual"] = row["observed"] - row["predicted_mu"]
        if roi in logp:
            row["logp"] = logp[roi].to_numpy()
        rows.append(row)
    long_df = pd.concat(rows, ignore_index=True)

    long_path = out_dir / "predictions_long.csv"
    long_df.to_csv(long_path, index=False)

    wide_parts = [id_df.copy()]
    for value_col, suffix in (
        ("predicted_mu", "mu"),
        ("z_score", "z"),
        ("residual", "resid"),
        ("observed", "observed"),
    ):
        part = long_df.pivot_table(
            index=pivot_index,
            columns="roi",
            values=value_col,
            aggfunc="first",
        ).reset_index()
        part.columns = [
            c if c in pivot_index else f"{c}__{suffix}"
            for c in part.columns
        ]
        wide_parts.append(part)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on=pivot_index, how="left")
    wide_path = out_dir / "predictions_wide.csv"
    wide.to_csv(wide_path, index=False)

    summary = pd.DataFrame(
        {
            "model_dir": [str(model_dir)],
            "n_subjects": [len(df)],
            "n_rois": [len(response_vars)],
            "n_missing_response_columns": [len(missing_responses)],
        }
    )
    summary_path = out_dir / "predict_summary.csv"
    summary.to_csv(summary_path, index=False)
    return {"long": long_path, "wide": wide_path, "summary": summary_path}


def infer_basis_params(model, nknots: int | None, degree: int | None, heteroskedastic: bool | None):
    template = model.template_regression_model
    basis = template.basis_function_mean
    inferred_degree = degree if degree is not None else int(getattr(basis, "degree", 3))
    if nknots is not None:
        inferred_nknots = nknots
    elif hasattr(basis, "nknots") and getattr(basis, "nknots") is not None:
        inferred_nknots = int(getattr(basis, "nknots"))
    elif hasattr(basis, "knots"):
        inferred_nknots = max(int(len(getattr(basis, "knots"))) - 2, 1)
    else:
        inferred_nknots = 3
    inferred_hetero = (
        heteroskedastic
        if heteroskedastic is not None
        else bool(getattr(template, "heteroskedastic", False))
    )
    return inferred_nknots, inferred_degree, inferred_hetero, getattr(template, "warp_name", None)


def fine_tune_model(
    model_dir: Path,
    csv_path: Path,
    out_dir: Path,
    nknots: int | None = None,
    degree: int | None = None,
    heteroskedastic: bool | None = None,
    min_n: int = 30,
    saveplots: bool = False,
) -> pd.DataFrame:
    from pcntoolkit import BLR, NormativeModel, NormData

    source = _load_normative_model(Path(model_dir))
    df = read_clean_csv(Path(csv_path))
    covariates = list(source.covariates)
    batch_effects = _batch_effects(source)
    response_vars = [roi for roi in source.response_vars if roi in df.columns]
    if not response_vars:
        raise ValueError("No model response variables are present in the fine-tune CSV")
    df = filter_required_rows(df, covariates, batch_effects, response_vars)
    if len(df) < min_n:
        raise ValueError(f"Not enough rows for fine-tuning: {len(df)} < {min_n}")

    use_nknots, use_degree, use_hetero, warp_name = infer_basis_params(
        source, nknots, degree, heteroskedastic
    )
    basis = create_bspline_basis(df[covariates[0]].values, use_nknots, use_degree, True)
    constant_covariates = [
        col for col in covariates
        if pd.to_numeric(df[col], errors="coerce").nunique(dropna=True) <= 1
    ]
    inscaler = "none" if constant_covariates else "standardize"
    blr = BLR(
        name="finetuned_blr",
        basis_function_mean=basis,
        fixed_effect=True,
        heteroskedastic=use_hetero,
        warp_name=warp_name,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    norm_data = NormData.from_dataframe(
        name="finetune",
        dataframe=df,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )
    model = NormativeModel(
        template_regression_model=blr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=saveplots,
        save_dir=str(out_dir),
        inscaler=inscaler,
        outscaler="standardize",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_predict(norm_data, norm_data)

    y_true = np.asarray(norm_data.Y.to_numpy(), dtype=np.float64)
    y_pred = np.asarray(result.Yhat.to_numpy(), dtype=np.float64)
    rows = []
    for idx, roi in enumerate(response_vars):
        rows.append({"roi": roi, **calculate_metrics(y_true[:, idx], y_pred[:, idx])})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "fine_tune_metrics_by_roi.csv", index=False)
    with open(out_dir / "fine_tune_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_model_dir": str(model_dir),
                "csv_path": str(csv_path),
                "n_samples": len(df),
                "n_response_vars": len(response_vars),
                "nknots": use_nknots,
                "degree": use_degree,
                "heteroskedastic": use_hetero,
                "inscaler": inscaler,
                "constant_covariates": constant_covariates,
            },
            f,
            indent=2,
        )
    return metrics


def model_template(model_dir: Path) -> list[str]:
    model = _load_normative_model(Path(model_dir))
    cols = ["subject_id"] + list(model.covariates) + _batch_effects(model) + list(model.response_vars)
    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out
