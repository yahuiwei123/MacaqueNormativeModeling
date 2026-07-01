from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .data import DatasetSpec, filter_required_rows, read_clean_csv
from .metrics import calculate_metrics


os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-pcntoolkit")


DEFAULT_PARAM_GRID = {
    "nknots": [2, 3, 4, 5],
    "degree": [2, 3],
    "heteroskedastic": [True, False],
    "warp_name": [None],
    "adaptive_knots": [True, False],
}


def _pcn():
    from pcntoolkit import BLR, BsplineBasisFunction, NormativeModel, NormData

    return BLR, BsplineBasisFunction, NormativeModel, NormData


def normalize_param_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def create_bspline_basis(age_data, nknots: int, degree: int, adaptive_knots: bool = True):
    _, BsplineBasisFunction, _, _ = _pcn()
    age_data = np.asarray(age_data, dtype=np.float64)
    if adaptive_knots:
        age_min = age_data.min()
        age_max = age_data.max()
        age_range = age_max - age_min
        knots = np.quantile(age_data, np.linspace(0, 1, int(nknots) + 2))
        knots[0] = age_min - 0.05 * age_range
        knots[-1] = age_max + 0.05 * age_range
        return BsplineBasisFunction(degree=int(degree), knots=knots)
    return BsplineBasisFunction(degree=int(degree), nknots=int(nknots))


def _array_from_result_yhat(result):
    if isinstance(result, tuple) and len(result) >= 1:
        yhat = result[0]
        return np.asarray(yhat.to_numpy(), dtype=np.float64)
    if hasattr(result, "Yhat"):
        return np.asarray(result.Yhat.to_numpy(), dtype=np.float64)
    if isinstance(result, dict) and "Yhat" in result:
        return np.asarray(result["Yhat"].to_numpy(), dtype=np.float64)
    if "Yhat" in result:
        return np.asarray(result["Yhat"].to_numpy(), dtype=np.float64)
    raise ValueError(f"Model fitting returned no Yhat: {type(result)}")


def cross_validation_optimize(
    data: pd.DataFrame,
    covariates: list[str],
    batch_effects: list[str],
    response_vars: list[str],
    n_folds: int = 5,
    param_grid: dict[str, list[Any]] | None = None,
    random_state: int = 42,
) -> tuple[dict[str, Any], pd.DataFrame]:
    BLR, _, NormativeModel, NormData = _pcn()
    grid = param_grid or DEFAULT_PARAM_GRID
    names = list(grid)
    combos = list(itertools.product(*[grid[name] for name in names]))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = []

    print(f"CV: {n_folds} folds, {len(combos)} parameter combinations")
    for idx, values in enumerate(combos, start=1):
        params = dict(zip(names, values))
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(data), start=1):
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            try:
                norm_train = NormData.from_dataframe(
                    name=f"cv_train_{fold}",
                    dataframe=train_fold,
                    covariates=covariates,
                    batch_effects=batch_effects,
                    response_vars=response_vars,
                )
                norm_val = NormData.from_dataframe(
                    name=f"cv_val_{fold}",
                    dataframe=val_fold,
                    covariates=covariates,
                    batch_effects=batch_effects,
                    response_vars=response_vars,
                )
                basis = create_bspline_basis(
                    train_fold[covariates[0]].values,
                    params["nknots"],
                    params["degree"],
                    params["adaptive_knots"],
                )
                blr = BLR(
                    name="cv_template",
                    basis_function_mean=basis,
                    fixed_effect=True,
                    heteroskedastic=params["heteroskedastic"],
                    warp_name=params["warp_name"],
                )
                model = NormativeModel(
                    template_regression_model=blr,
                    savemodel=False,
                    evaluate_model=False,
                    saveresults=False,
                    saveplots=False,
                    inscaler="standardize",
                    outscaler="standardize",
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = model.fit_predict(norm_train, norm_val)
                y_pred = _array_from_result_yhat(result)
                y_true = np.asarray(norm_val.Y.to_numpy(), dtype=np.float64)
                if y_pred.ndim == 1:
                    y_pred = y_pred.reshape(-1, 1)
                if y_true.ndim == 1:
                    y_true = y_true.reshape(-1, 1)
                expv = [
                    calculate_metrics(y_true[:, roi_idx], y_pred[:, roi_idx])["expv"]
                    for roi_idx in range(y_pred.shape[1])
                ]
                fold_scores.append(np.nanmean(expv))
            except Exception as exc:
                print(f"  Fold {fold} failed for {params}: {exc}")
                fold_scores.append(np.nan)

        row = {
            **params,
            "mean_expv": float(np.nanmean(fold_scores)),
            "std_expv": float(np.nanstd(fold_scores)),
            "fold_scores": json.dumps([None if np.isnan(x) else float(x) for x in fold_scores]),
        }
        print(f"  [{idx}/{len(combos)}] EXPV={row['mean_expv']:.4f} params={params}")
        results.append(row)

    cv_df = pd.DataFrame(results).sort_values("mean_expv", ascending=False).reset_index(drop=True)
    best = cv_df.iloc[0].to_dict()
    for key in ("mean_expv", "std_expv", "fold_scores"):
        best.pop(key, None)
    best = {key: normalize_param_value(value) for key, value in best.items()}
    return best, cv_df


def _make_model(params: dict[str, Any], age_values):
    BLR, _, NormativeModel, _ = _pcn()
    basis = create_bspline_basis(
        age_values,
        int(params["nknots"]),
        int(params["degree"]),
        bool(params.get("adaptive_knots", True)),
    )
    blr = BLR(
        name="template",
        basis_function_mean=basis,
        fixed_effect=True,
        heteroskedastic=bool(params["heteroskedastic"]),
        warp_name=normalize_param_value(params.get("warp_name")),
    )
    return NormativeModel, blr


def train_cv_model(
    spec: DatasetSpec,
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    saveplots: bool = False,
) -> dict[str, Any]:
    _, _, NormativeModel, NormData = _pcn()
    df = read_clean_csv(spec.data_path)
    df = filter_required_rows(df, spec.covariates, spec.batch_effects, spec.response_vars)
    train, test = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    response_vars = list(spec.response_vars)
    covariates = list(spec.covariates)
    batch_effects = list(spec.batch_effects)

    spec.save_dir.mkdir(parents=True, exist_ok=True)
    best_params, cv_df = cross_validation_optimize(
        train,
        covariates,
        batch_effects,
        response_vars,
        n_folds=n_folds,
        random_state=random_state,
    )
    cv_df.to_csv(spec.save_dir / "cross_validation_results.csv", index=False)

    norm_train = NormData.from_dataframe(
        name="train",
        dataframe=train,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )
    norm_test = NormData.from_dataframe(
        name="test",
        dataframe=test,
        covariates=covariates,
        batch_effects=batch_effects,
        response_vars=response_vars,
    )

    _, blr = _make_model(best_params, train[covariates[0]].values)
    model = NormativeModel(
        template_regression_model=blr,
        savemodel=True,
        evaluate_model=True,
        saveresults=True,
        saveplots=saveplots,
        save_dir=str(spec.save_dir),
        inscaler="standardize",
        outscaler="standardize",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_predict(norm_train, norm_test)

    y_pred = _array_from_result_yhat(result)
    y_true = np.asarray(norm_test.Y.to_numpy(), dtype=np.float64)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    rows = []
    for idx, roi in enumerate(response_vars):
        rows.append(
            {
                "model_type": "global" if spec.is_global else "local",
                "atlas": spec.atlas,
                "hemi": spec.hemi,
                "metric": spec.metric,
                "roi": roi,
                **best_params,
                **calculate_metrics(y_true[:, idx], y_pred[:, idx]),
            }
        )
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(spec.save_dir / "test_set_results_by_roi.csv", index=False)
    if spec.is_global:
        metrics_df.to_csv(spec.save_dir / "test_set_results.csv", index=False)

    summary = {
        "label": spec.label,
        "n_samples": len(df),
        "n_train": len(train),
        "n_test": len(test),
        "n_response_vars": len(response_vars),
        "best_params": best_params,
        "mean_expv": float(metrics_df["expv"].mean()),
    }
    with open(spec.save_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def params_from_row(row: pd.Series) -> dict[str, Any]:
    return {
        "nknots": int(row["nknots"]),
        "degree": int(row["degree"]),
        "heteroskedastic": bool(row["heteroskedastic"]),
        "warp_name": normalize_param_value(row.get("warp_name")),
        "adaptive_knots": bool(row.get("adaptive_knots", True)),
    }


def train_full_model(
    spec: DatasetSpec,
    params: dict[str, Any] | pd.Series,
    saveplots: bool = False,
    evaluate_model: bool = True,
) -> dict[str, Any]:
    _, _, NormativeModel, NormData = _pcn()
    if isinstance(params, pd.Series):
        params = params_from_row(params)
    else:
        params = {key: normalize_param_value(value) for key, value in params.items()}

    df = read_clean_csv(spec.data_path)
    df = filter_required_rows(df, spec.covariates, spec.batch_effects, spec.response_vars)
    save_dir = spec.save_dir / "full_data_model"
    save_dir.mkdir(parents=True, exist_ok=True)

    norm_data = NormData.from_dataframe(
        name=f"full_{spec.atlas}_{spec.hemi}_{spec.metric}",
        dataframe=df,
        covariates=list(spec.covariates),
        batch_effects=list(spec.batch_effects),
        response_vars=list(spec.response_vars),
    )
    _, blr = _make_model(params, df[spec.covariates[0]].values)
    model = NormativeModel(
        template_regression_model=blr,
        savemodel=True,
        evaluate_model=evaluate_model,
        saveresults=evaluate_model,
        saveplots=saveplots,
        save_dir=str(save_dir),
        inscaler="standardize",
        outscaler="standardize",
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(norm_data)
    except Exception as exc:
        print(f"Fit finished with post-processing error, saving core model: {exc}")
        model.save(str(save_dir))

    summary = {
        "label": spec.label,
        "n_samples": len(df),
        "n_response_vars": len(spec.response_vars),
        "params": params,
        "save_dir": str(save_dir),
    }
    with open(save_dir / "full_training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def analyze_cv_results(save_base: Path) -> pd.DataFrame:
    rows = []
    save_base = Path(save_base)
    for csv_path in save_base.rglob("cross_validation_results.csv"):
        parts = csv_path.parts
        if "subcort" in parts:
            idx = parts.index("subcort")
            atlas = "subcortical"
            hemi = parts[idx + 1]
            metric = parts[idx + 2]
            if metric.startswith("global_global_"):
                metric = metric.replace("global_global_", "global_", 1)
        else:
            idx = parts.index("save_dir")
            atlas = parts[idx + 1]
            hemi = parts[idx + 2]
            metric = parts[idx + 3]
        df = pd.read_csv(csv_path)
        valid = df.dropna(subset=["mean_expv"])
        if valid.empty:
            continue
        best = valid.sort_values("mean_expv", ascending=False).iloc[0]
        rows.append(
            {
                "atlas": atlas,
                "hemi": hemi,
                "metric": metric,
                "nknots": int(best["nknots"]),
                "degree": int(best["degree"]),
                "heteroskedastic": bool(best["heteroskedastic"]),
                "warp_name": best["warp_name"] if pd.notna(best["warp_name"]) else None,
                "adaptive_knots": bool(best["adaptive_knots"]),
                "mean_expv": round(float(best["mean_expv"]), 4),
                "std_expv": round(float(best["std_expv"]), 4),
                "total_valid_params": len(valid),
                "total_params": len(df),
            }
        )
    out = pd.DataFrame(rows).sort_values(["atlas", "hemi", "metric"]).reset_index(drop=True)
    out.to_csv(save_base / "optimal_parameters_summary.csv", index=False, encoding="utf-8-sig")
    return out
