# Macaque Normative Modeling Toolkit

Bayesian Linear Regression (BLR) toolkit for macaque structural MRI normative
modeling. The pipeline harmonizes multi-site data with ComBat, runs 5-fold
cross-validation to select BLR hyperparameters, trains final full-data
normative models, fine-tunes pretrained models on fixed-format user data, and
predicts deviation z-scores for new subjects.

This branch reflects the current Macaque Normative Modeling workflow used with:

- `healthy_data/`: wide-format cortical and subcortical CSV tables
- `save_dir/`: fitted pcntoolkit BLR models and CV summaries
- `full_data_model/`: final models refit on all available data after CV model
  selection

The current model directory contains **7,468 pcntoolkit ROI-level model files**:
**3,734 split/CV models** plus **3,734 final full-data models**. Inference should
normally use the `full_data_model` version.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Step 1: Inspect Data and Models](#step-1-inspect-data-and-models)
  - [Step 2: ComBat Harmonization](#step-2-combat-harmonization)
  - [Step 3: Cross-Validation Training](#step-3-cross-validation-training)
  - [Step 4: Full-Data Training](#step-4-full-data-training)
  - [Step 5: Prediction](#step-5-prediction)
  - [Step 6: Fine-Tuning](#step-6-fine-tuning)
- [Model Directory Semantics](#model-directory-semantics)
- [Pretrained Model Inventory](#pretrained-model-inventory)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [Module Reference](#module-reference)
- [Citation](#citation)

## Pipeline Overview

```text
Raw multi-site healthy_data CSVs
  |
  +- Step 1: ComBat harmonization (monkey_normative.combat)
  |   Remove site effects while preserving age-related trends
  |
  +- Step 2: CV training (monkey_normative.blr)
  |   5-fold CV -> best hyperparameters -> train/test split model
  |
  +- Step 3: CV summary
  |   cross_validation_results.csv -> optimal_parameters_summary.csv
  |
  +- Step 4: full_data_model training
  |   Refit selected BLR specification on all available subjects
  |
  +- Step 5: prediction
  |   full_data_model + fixed-format CSV -> mu, residual, z-score, logp
  |
  +- Step 6: fine-tuning
      Refit a pretrained model on a user cohort with matching columns
```

**Model specification**: BLR with B-spline basis functions, fixed effects for
covariates, batch effects for sex/breed, optional heteroskedastic variance, and
optional warping. The CV grid currently evaluates:

| Parameter | Values |
|-----------|--------|
| `nknots` | 2, 3, 4, 5 |
| `degree` | 2, 3 |
| `heteroskedastic` | True, False |
| `adaptive_knots` | True, False |
| `warp_name` | None |

## Installation

```bash
git clone https://github.com/yahuiwei123/MacaqueNormativeModeling.git
cd MacaqueNormativeModeling
pip install -e .
```

For BLR training, prediction, and fine-tuning, install with the BLR extra:

```bash
pip install -e ".[blr]"
```

For ComBat harmonization:

```bash
pip install -e ".[combat]"
```

Or install both optional dependency groups:

```bash
pip install -e ".[blr,combat]"
```

**Key dependencies**:

| Package | Purpose | Notes |
|---------|---------|-------|
| `pcntoolkit` | BLR training, loading, prediction, z-scores | Python 3.10-3.12 recommended |
| `neuroHarmonize` | ComBat harmonization | Required only for `combat` |
| `scikit-learn` | train/test split and K-fold CV | Core dependency |
| `pandas`, `numpy`, `scipy` | Table processing and metrics | Core dependencies |

Example conda environment:

```bash
conda create -n pcnnorm python=3.11
conda activate pcnnorm
pip install -e ".[blr,combat]"
```

## Quick Start

Inspect the current dataset/model inventory:

```bash
monkey-norm inspect --phase all
```

Generate an input template from a trained final model:

```bash
monkey-norm template \
  --atlas MBNA124 \
  --hemi L \
  --metric thickness \
  --out-csv user_input_template.csv
```

Predict deviations using the final `full_data_model`:

```bash
monkey-norm predict \
  --atlas MBNA124 \
  --hemi L \
  --metric thickness \
  --csv user_input.csv \
  --out-dir results/predictions/MBNA124_L_thickness
```

Fine-tune a final model on a user cohort:

```bash
monkey-norm fine-tune \
  --atlas MBNA124 \
  --hemi L \
  --metric thickness \
  --csv user_input.csv \
  --out-dir results/finetuned/MBNA124_L_thickness
```

Predict with the fine-tuned model:

```bash
monkey-norm predict \
  --model-dir results/finetuned/MBNA124_L_thickness \
  --csv user_input.csv \
  --out-dir results/predictions/finetuned_MBNA124_L_thickness
```

## Data Format

Input CSV files are wide-format, one row per imaging session or subject record.
For prediction and fine-tuning, use the exact columns expected by the selected
model. The safest way to obtain the header is:

```bash
monkey-norm template --atlas Modalities --hemi R --metric thickness --out-csv template.csv
```

Required columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `subject_id` | string | Yes | Subject/session identifier. Repeated IDs are allowed. |
| `age` | numeric | Yes | Primary age covariate. |
| `sex` | string | Yes | Batch effect used by the BLR model. |
| `breed` | string | Yes | Batch effect used by the BLR model. |
| `site` | string | For ComBat | Scanner/site label used during harmonization. |
| `global_*` | numeric | Model-dependent | Global covariates or global response variables. |
| `<ROI>` | numeric | For z-scores | ROI measurement columns matching the model response variables. |

Example:

```csv
subject_id,age,sex,breed,AUDITORY,COGNITIVE,VISION
sub-001,3.5,F,Macaca mulatta,1.56,1.71,1.62
sub-002,8.2,M,Macaca fascicularis,1.49,1.68,1.57
```

If ROI observations are missing, `predict` can still produce model means with
`--allow-missing-responses`, but deviation z-scores require observed ROI values.

Prediction outputs:

- `predictions_long.csv`: `subject_id, roi, observed, predicted_mu, residual, z_score, logp`
- `predictions_wide.csv`: one row per input row with `ROI__mu`, `ROI__z`,
  `ROI__resid`, and `ROI__observed` columns
- `predict_summary.csv`: model path and prediction counts

The wide output includes `row_id` so repeated or longitudinal `subject_id`
records are not collapsed.

## Usage

### Step 1: Inspect Data and Models

Check expected model counts from the data and compare against saved model files:

```bash
monkey-norm inspect --phase all
```

Inspect one subset:

```bash
monkey-norm inspect --phase full --atlas subcortical --hemi R
```

Write a machine-readable audit table:

```bash
monkey-norm inspect --phase all --out-csv model_inventory_audit.csv
```

### Step 2: ComBat Harmonization

ComBat uses `site` as the batch variable and preserves age as a smooth biological
term.

```bash
# Harmonize every eligible CSV under healthy_data
monkey-norm combat --base-dir /path/to/healthy_data

# Harmonize a single CSV
monkey-norm combat \
  --csv /path/to/healthy_data/cort/MBNA124/L/thickness.csv
```

By default, outputs are written to a `harmonized/` subdirectory next to each CSV.

### Step 3: Cross-Validation Training

Run 5-fold CV and train/test split evaluation for one model group:

```bash
monkey-norm train-cv \
  --atlas MBNA124 \
  --hemi L \
  --metric thickness \
  --n-folds 5
```

Run all CV tasks:

```bash
monkey-norm train-cv --all
```

CV/training outputs:

```text
save_dir/{atlas}/{hemi}/{metric}/
├── model/                          # train/test split pcntoolkit model
│   ├── normative_model.json
│   └── {ROI}/regression_model.json
├── cross_validation_results.csv    # all parameter combinations
├── test_set_results_by_roi.csv     # test-set metrics
└── training_summary.json
```

After CV tasks finish, summarize best parameters:

```bash
monkey-norm analyze-cv --save-dir /path/to/save_dir
```

This writes:

```text
save_dir/optimal_parameters_summary.csv
```

### Step 4: Full-Data Training

Train final models from `optimal_parameters_summary.csv`:

```bash
monkey-norm train-full \
  --atlas MBNA124 \
  --hemi L \
  --metric thickness
```

Train every row in the parameter summary:

```bash
monkey-norm train-full --all
```

Final models are saved under:

```text
save_dir/{atlas}/{hemi}/{metric}/full_data_model/
├── model/
│   ├── normative_model.json
│   └── {ROI}/regression_model.json
└── full_training_summary.json
```

### Step 5: Prediction

Use final full-data models for prediction:

```bash
monkey-norm predict \
  --atlas M129 \
  --hemi R \
  --metric sulc \
  --csv new_subjects.csv \
  --out-dir results/predict_M129_R_sulc
```

Predict selected ROIs only:

```bash
monkey-norm predict \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --csv new_subjects.csv \
  --out-dir results/predict_modalities_R_thickness \
  --rois AUDITORY,VISION
```

Use an explicit model directory:

```bash
monkey-norm predict \
  --model-dir /path/to/MBNA124/L/thickness/full_data_model \
  --csv new_subjects.csv \
  --out-dir results/predict_explicit_model
```

Interpretation: large absolute z-scores indicate deviations from the normative
distribution. A common two-sided reference threshold is `|z_score| > 1.96`.

### Step 6: Fine-Tuning

Fine-tune a pretrained full-data model on user data with matching columns:

```bash
monkey-norm fine-tune \
  --atlas M132 \
  --hemi L \
  --metric cortvol \
  --csv user_cohort.csv \
  --out-dir results/finetuned/M132_L_cortvol
```

Optional overrides:

```bash
monkey-norm fine-tune \
  --atlas M132 \
  --hemi L \
  --metric cortvol \
  --csv user_cohort.csv \
  --out-dir results/finetuned/M132_L_cortvol \
  --nknots 4 \
  --degree 3 \
  --heteroskedastic true
```

Fine-tuning saves a complete pcntoolkit model that can be passed directly to
`monkey-norm predict --model-dir`.

## Model Directory Semantics

The repository distinguishes two saved model types:

```text
save_dir/{atlas}/{hemi}/{metric}/model/
```

This is the **split/CV model**. It is produced during `train-cv`: the script
performs 5-fold CV on the training portion, selects the best parameters, fits on
the train split, and evaluates on a held-out test split.

```text
save_dir/{atlas}/{hemi}/{metric}/full_data_model/model/
```

This is the **final inference model**. It uses the selected parameters from
`optimal_parameters_summary.csv` and refits on all available data.

For user prediction and deviation scoring, prefer `full_data_model`.

Compatibility note: the legacy CV code includes the corresponding `global_*`
metric as an additional covariate for cortical local ROI models. The legacy
full-data training code uses `age` only by default. This toolkit preserves that
behavior:

- `train-cv` includes local global covariates by default; use
  `--no-global-covariate` to disable.
- `train-full` does not include local global covariates by default; use
  `--include-global-covariate` to enable.

## Pretrained Model Inventory

The current `save_dir` inventory is:

```text
save_dir/
├── MBNA124/{L,R}/{cortvol,curvature,sulc,thickness,area,global_*}/
├── M129/{L,R}/{cortvol,curvature,sulc,thickness,area,global_*}/
├── M132/{L,R}/{cortvol,curvature,sulc,thickness,area,global_*}/
├── Modalities/{L,R}/{cortvol,curvature,sulc,thickness,area,global_*}/
├── subcort/{L,R}/{volume,global_estimated_ICV_mm3,global_total_subcortical_vol_mm3}/
└── optimal_parameters_summary.csv
```

Full-data model counts:

| Atlas | Full-data ROI-level models |
|-------|----------------------------|
| MBNA124 | 1,498 |
| M129 | 970 |
| M132 | 1,114 |
| Modalities | 118 |
| Subcortical | 34 |
| **Total full_data_model** | **3,734** |

Including split/CV models, the current saved model count is:

| Model phase | ROI-level model files |
|-------------|-----------------------|
| split/CV `model/` | 3,734 |
| final `full_data_model/model/` | 3,734 |
| **Total** | **7,468** |

Known legacy path issue: two right-hemisphere subcortical split global models
were saved under `global_global_*` directories in older runs. The final
`full_data_model` paths are correct and complete.

## Fine-Tuning Guide

### When to Fine-Tune vs Train From Scratch

| Scenario | Recommendation | Rationale |
|----------|----------------|-----------|
| New cohort with matching atlas/metric columns | Fine-tune | Keeps model structure while adapting parameters |
| New scanner/site with enough matched data | Fine-tune | Useful after local harmonization or site shift |
| Similar macaque population, small sample | Fine-tune | More stable than running CV from scratch |
| Very different age distribution | Fine-tune with knot override or retrain | Age basis may need more flexibility |
| New atlas, metric, or incompatible ROI names | Train from scratch | Response variables do not match pretrained models |
| Large healthy dataset | Train CV, then train full | Best for a new final model |

### Strategy by Dataset Size

Small datasets (`n < 50`): keep the original structure.

```bash
monkey-norm fine-tune \
  --model-dir pretrained/MBNA124/L/thickness/full_data_model \
  --csv small_dataset.csv \
  --out-dir finetuned_small
```

Medium datasets (`50 <= n < 200`): consider explicit structure overrides.

```bash
monkey-norm fine-tune \
  --model-dir pretrained/MBNA124/L/thickness/full_data_model \
  --csv medium_dataset.csv \
  --out-dir finetuned_medium \
  --nknots 4 \
  --heteroskedastic true
```

Large datasets (`n >= 200`): run full CV, then refit on all data.

```bash
monkey-norm train-cv --atlas MBNA124 --hemi L --metric thickness
monkey-norm analyze-cv
monkey-norm train-full --atlas MBNA124 --hemi L --metric thickness
```

### Evaluating Fine-Tuning Quality

```bash
monkey-norm predict \
  --model-dir finetuned_medium \
  --csv medium_dataset.csv \
  --out-dir eval_finetuned

python -c "
import pandas as pd
df = pd.read_csv('eval_finetuned/predictions_long.csv')
print(f'Mean z: {df[\"z_score\"].mean():.3f}')
print(f'Std z:  {df[\"z_score\"].std():.3f}')
print(f'|z|>1.96: {(df[\"z_score\"].abs() > 1.96).mean()*100:.1f}%')
"
```

Healthy samples should usually have z-scores centered near zero with scale near
one, though exact behavior depends on cohort composition and harmonization.

## Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `monkey_normative.cli` | Command-line interface | `main()` |
| `monkey_normative.data` | Dataset specs, ROI selection, model audits | `iter_cv_specs()`, `iter_full_specs_from_params()` |
| `monkey_normative.combat` | ComBat harmonization | `harmonize_csv()`, `harmonize_many()` |
| `monkey_normative.blr` | CV training, full-data training, CV summary | `train_cv_model()`, `train_full_model()`, `analyze_cv_results()` |
| `monkey_normative.predict` | Template export, prediction, fine-tuning | `predict_deviations()`, `fine_tune_model()`, `model_template()` |
| `monkey_normative.metrics` | Shared evaluation metrics | `calculate_metrics()` |

## Citation

If you use this toolkit or the pretrained models, please cite:

> Wei, Y. et al. MacaSurfer: unified surface-volume mapping of the macaque brain
> across the lifespan. 2026.06.14.732101 Preprint at
> https://doi.org/10.64898/2026.06.14.732101 (2026).

> Fraza, C. et al. (2021). Warped Bayesian linear regression for normative
> modelling of big data. NeuroImage, 245, 118715.

> Fortin, J.-P. et al. (2018). Harmonization of cortical thickness measurements
> across scanners and sites. NeuroImage, 167, 104-120.
