from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

from .blr import analyze_cv_results, train_cv_model, train_full_model
from .combat import find_csv_files, harmonize_csv, harmonize_many
from .constants import DEFAULT_HEALTHY_DATA_DIR, DEFAULT_SAVE_DIR
from .data import (
    DatasetSpec,
    cortical_global_spec,
    cortical_local_spec,
    iter_cv_specs,
    iter_full_specs_from_params,
    spec_audit_rows,
    subcort_global_spec,
    subcort_local_spec,
)
from .predict import fine_tune_model, model_template, predict_deviations, resolve_model_dir


def path_arg(value: str) -> Path:
    return Path(value).expanduser().resolve()


def add_common_paths(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--base-dir", type=path_arg, default=DEFAULT_HEALTHY_DATA_DIR)
    ap.add_argument("--save-dir", type=path_arg, default=DEFAULT_SAVE_DIR)


def filter_specs(
    specs: list[DatasetSpec],
    atlas: str | None,
    hemi: str | None,
    metric: str | None,
) -> list[DatasetSpec]:
    out = specs
    if atlas:
        out = [s for s in out if s.atlas == atlas]
    if hemi:
        out = [s for s in out if s.hemi == hemi]
    if metric:
        out = [s for s in out if s.metric == metric]
    return out


def command_inspect(args) -> int:
    rows = []
    if args.phase in ("cv", "all"):
        specs = iter_cv_specs(
            args.base_dir,
            args.save_dir,
            use_harmonized=not args.no_harmonized,
            include_global_models=not args.no_global_models,
            include_global_covariate=not args.no_global_covariate,
            min_n=args.min_n,
        )
        rows.extend(spec_audit_rows(filter_specs(specs, args.atlas, args.hemi, args.metric), full=False))
    if args.phase in ("full", "all"):
        params = args.params or args.save_dir / "optimal_parameters_summary.csv"
        if params.exists():
            specs_rows = iter_full_specs_from_params(
                params,
                args.base_dir,
                args.save_dir,
                use_harmonized=args.full_harmonized,
                include_global_covariate=args.full_include_global_covariate,
                min_n=args.full_min_n,
            )
            specs = [spec for spec, _ in specs_rows]
            rows.extend(spec_audit_rows(filter_specs(specs, args.atlas, args.hemi, args.metric), full=True))
        else:
            print(f"Skipping full inspection, params file not found: {params}")
    df = pd.DataFrame(rows)
    if df.empty:
        print("No matching specs found")
        return 1
    print(df[["phase", "atlas", "hemi", "metric", "expected_models", "actual_models", "n_samples"]].to_string(index=False))
    mismatches = df[df["expected_models"] != df["actual_models"]]
    print(
        f"\nSummary: combos={len(df)}, expected_models={df['expected_models'].sum()}, "
        f"actual_models={df['actual_models'].sum()}, mismatches={len(mismatches)}"
    )
    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"Wrote: {args.out_csv}")
    return 0


def command_combat(args) -> int:
    if args.csv:
        out = harmonize_csv(args.csv, args.output_dir, save_model=not args.no_save_model)
        return 0 if out else 1
    files = find_csv_files(args.base_dir, args.pattern)
    outputs = harmonize_many(
        files,
        input_root=args.base_dir,
        output_root=args.output_dir,
        save_model=not args.no_save_model,
    )
    print(f"Harmonized {len(outputs)}/{len(files)} files")
    return 0


def command_train_cv(args) -> int:
    specs = iter_cv_specs(
        args.base_dir,
        args.save_dir,
        use_harmonized=not args.no_harmonized,
        include_global_models=not args.no_global_models,
        include_global_covariate=not args.no_global_covariate,
        min_n=args.min_n,
    )
    specs = filter_specs(specs, args.atlas, args.hemi, args.metric)
    if not args.all and len(specs) != 1:
        print("Select exactly one spec, or pass --all.")
        print(f"Matching specs: {[s.label for s in specs[:20]]}")
        return 1
    for spec in specs:
        print(f"\nTraining CV model: {spec.label}")
        train_cv_model(
            spec,
            n_folds=args.n_folds,
            test_size=args.test_size,
            random_state=args.random_state,
            saveplots=args.saveplots,
        )
    return 0


def command_analyze_cv(args) -> int:
    df = analyze_cv_results(args.save_dir)
    print(f"Analyzed {len(df)} CV result files")
    print(f"Wrote: {args.save_dir / 'optimal_parameters_summary.csv'}")
    return 0


def command_train_full(args) -> int:
    params = args.params or args.save_dir / "optimal_parameters_summary.csv"
    specs_rows = iter_full_specs_from_params(
        params,
        args.base_dir,
        args.save_dir,
        use_harmonized=args.use_harmonized,
        include_global_covariate=args.include_global_covariate,
        min_n=args.min_n,
    )
    specs_rows = [
        (spec, row)
        for spec, row in specs_rows
        if (not args.atlas or spec.atlas == args.atlas)
        and (not args.hemi or spec.hemi == args.hemi)
        and (not args.metric or spec.metric == args.metric)
    ]
    if not args.all and len(specs_rows) != 1:
        print("Select exactly one full model, or pass --all.")
        print(f"Matching specs: {[spec.label for spec, _ in specs_rows[:20]]}")
        return 1
    for spec, row in specs_rows:
        print(f"\nTraining full-data model: {spec.label}")
        train_full_model(
            spec,
            row,
            saveplots=args.saveplots,
            evaluate_model=not args.no_evaluate,
        )
    return 0


def resolved_model_arg(args) -> Path:
    return resolve_model_dir(
        model_dir=args.model_dir,
        save_base=args.save_dir,
        atlas=args.atlas,
        hemi=args.hemi,
        metric=args.metric,
        full=not args.no_full,
    )


def command_predict(args) -> int:
    model_dir = resolved_model_arg(args)
    rois = [r.strip() for r in args.rois.split(",") if r.strip()] or None
    paths = predict_deviations(
        model_dir,
        args.csv,
        args.out_dir,
        roi_names=rois,
        allow_missing_responses=args.allow_missing_responses,
    )
    print(f"Wrote: {paths['long']}")
    print(f"Wrote: {paths['wide']}")
    return 0


def command_fine_tune(args) -> int:
    model_dir = resolved_model_arg(args)
    metrics = fine_tune_model(
        model_dir,
        args.csv,
        args.out_dir,
        nknots=args.nknots,
        degree=args.degree,
        heteroskedastic=args.heteroskedastic,
        min_n=args.min_n,
        saveplots=args.saveplots,
    )
    print(f"Fine-tuned {len(metrics)} response variables")
    print(f"Wrote: {args.out_dir}")
    return 0


def command_template(args) -> int:
    model_dir = resolved_model_arg(args)
    cols = model_template(model_dir)
    if args.out_csv:
        pd.DataFrame(columns=cols).to_csv(args.out_csv, index=False)
        print(f"Wrote: {args.out_csv}")
    else:
        print(",".join(cols))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="monkey-norm", description="Macaque BLR normative modeling pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect = sub.add_parser("inspect", help="Inspect expected data/model counts")
    add_common_paths(inspect)
    inspect.add_argument("--phase", choices=["cv", "full", "all"], default="all")
    inspect.add_argument("--atlas")
    inspect.add_argument("--hemi")
    inspect.add_argument("--metric")
    inspect.add_argument("--params", type=path_arg)
    inspect.add_argument("--out-csv", type=path_arg)
    inspect.add_argument("--no-harmonized", action="store_true")
    inspect.add_argument("--no-global-models", action="store_true")
    inspect.add_argument("--no-global-covariate", action="store_true")
    inspect.add_argument("--min-n", type=int, default=30)
    inspect.add_argument("--full-harmonized", action="store_true")
    inspect.add_argument("--full-include-global-covariate", action="store_true")
    inspect.add_argument("--full-min-n", type=int, default=100)
    inspect.set_defaults(func=command_inspect)

    combat = sub.add_parser("combat", help="Apply ComBat harmonization")
    combat.add_argument("--base-dir", type=path_arg, default=DEFAULT_HEALTHY_DATA_DIR)
    combat.add_argument("--csv", type=path_arg)
    combat.add_argument("--output-dir", type=path_arg)
    combat.add_argument("--pattern", default="*.csv")
    combat.add_argument("--no-save-model", action="store_true")
    combat.set_defaults(func=command_combat)

    train_cv = sub.add_parser("train-cv", help="Run 5-fold CV and train/test BLR")
    add_common_paths(train_cv)
    train_cv.add_argument("--all", action="store_true")
    train_cv.add_argument("--atlas")
    train_cv.add_argument("--hemi")
    train_cv.add_argument("--metric")
    train_cv.add_argument("--no-harmonized", action="store_true")
    train_cv.add_argument("--no-global-models", action="store_true")
    train_cv.add_argument("--no-global-covariate", action="store_true")
    train_cv.add_argument("--min-n", type=int, default=30)
    train_cv.add_argument("--n-folds", type=int, default=5)
    train_cv.add_argument("--test-size", type=float, default=0.2)
    train_cv.add_argument("--random-state", type=int, default=42)
    train_cv.add_argument("--saveplots", action="store_true")
    train_cv.set_defaults(func=command_train_cv)

    analyze = sub.add_parser("analyze-cv", help="Build optimal_parameters_summary.csv")
    analyze.add_argument("--save-dir", type=path_arg, default=DEFAULT_SAVE_DIR)
    analyze.set_defaults(func=command_analyze_cv)

    full = sub.add_parser("train-full", help="Train full-data models from optimal CV params")
    add_common_paths(full)
    full.add_argument("--params", type=path_arg)
    full.add_argument("--all", action="store_true")
    full.add_argument("--atlas")
    full.add_argument("--hemi")
    full.add_argument("--metric")
    full.add_argument("--use-harmonized", action="store_true")
    full.add_argument("--include-global-covariate", action="store_true")
    full.add_argument("--min-n", type=int, default=100)
    full.add_argument("--saveplots", action="store_true")
    full.add_argument("--no-evaluate", action="store_true")
    full.set_defaults(func=command_train_full)

    for name, help_text, func in (
        ("predict", "Predict deviations with a trained full_data_model", command_predict),
        ("fine-tune", "Fine-tune a full_data_model on fixed-format user data", command_fine_tune),
        ("template", "Write the fixed-format CSV header expected by a model", command_template),
    ):
        ap = sub.add_parser(name, help=help_text)
        ap.add_argument("--model-dir", type=path_arg)
        ap.add_argument("--save-dir", type=path_arg, default=DEFAULT_SAVE_DIR)
        ap.add_argument("--atlas")
        ap.add_argument("--hemi")
        ap.add_argument("--metric")
        ap.add_argument("--no-full", action="store_true", help="Use split model instead of full_data_model")
        if name != "template":
            ap.add_argument("--csv", type=path_arg, required=True)
            ap.add_argument("--out-dir", type=path_arg, required=True)
        if name == "predict":
            ap.add_argument("--rois", default="")
            ap.add_argument("--allow-missing-responses", action="store_true")
        if name == "fine-tune":
            ap.add_argument("--nknots", type=int)
            ap.add_argument("--degree", type=int)
            ap.add_argument("--heteroskedastic", type=lambda x: x.lower() == "true")
            ap.add_argument("--min-n", type=int, default=30)
            ap.add_argument("--saveplots", action="store_true")
        if name == "template":
            ap.add_argument("--out-csv", type=path_arg)
        ap.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ModuleNotFoundError as exc:
        if "pcntoolkit" in str(exc):
            print("pcntoolkit is required for BLR train/predict/fine-tune commands.", file=sys.stderr)
            print("Use the pcnnorm environment or install pcntoolkit.", file=sys.stderr)
            return 2
        if "neuroHarmonize" in str(exc):
            print("neuroHarmonize is required for combat.", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
