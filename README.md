# GPR Normative Model

Gaussian Process Regression (GPR) toolkit for brain normative modeling. Trains normative models on healthy subjects, evaluates model fit via cross-validation, and predicts deviation scores (z-scores) for new subjects.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Training](#training)
  - [Cross-Validation](#cross-validation)
  - [Prediction](#prediction)
  - [Fine-Tuning](#fine-tuning)
- [Understanding GPR Hyperparameters](#understanding-gpr-hyperparameters)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [API Reference](#api-reference)

## Overview

The model uses a Gaussian Process with a **Constant × Matern + White Noise** kernel:

```
K(x, x') = σ² · k_matern(x, x'; ν, ℓ) + σ_n² · δ(x, x')
```

| Parameter | Role | Description |
|-----------|------|-------------|
| σ² (constant_value) | Signal variance | Overall magnitude of brain measure variation |
| ℓ (length_scale) | Spatial scale | How fast brain measures change with age |
| ν (nu) | Smoothness | Matern smoothness: 0.5=rough, 1.5=smooth, 2.5=very smooth |
| σ_n² (noise_level) | Noise | Measurement noise / unexplained variance |

The model uses `age` as the primary covariate (numeric) and `sex`, `site`, `breed` as categorical batch effects.

**Output** for each subject and brain region:
- `mu` — predicted normative value
- `std` — prediction uncertainty
- `resid` — residual (actual − predicted)
- `z` — z-score (resid / std); |z| > 1.96 indicates significant deviation

## Installation

```bash
# Clone or copy the project
cd gpr_normative_model

# Install dependencies
pip install -r requirements.txt

# Optional: install as a package
pip install -e .
```

**Requirements**: Python ≥ 3.10, numpy, pandas, scipy, scikit-learn ≥ 1.0, joblib

## Quick Start

```bash
# 1. Generate example data
python data/generate_example_data.py --out_dir data --n_subjects 200

# 2. Train models with a holdout test set
python -m gpr_normative.train \
    --csv data/example_brain_data.csv \
    --out_dir results/quickstart \
    --test_size 0.2 \
    --n_jobs 4

# 3. Evaluate with cross-validation
python -m gpr_normative.evaluate \
    --csv data/example_brain_data.csv \
    --out_dir results/cv \
    --n_folds 5 \
    --n_jobs 4

# 4. Predict on the same data (simulating new subjects)
python -m gpr_normative.predict \
    --in_path data/example_brain_data.csv \
    --model_dir results/quickstart/models \
    --out_dir results/predictions
```

Or use the Jupyter notebook: `notebooks/example_usage.ipynb`

## Data Format

Expected input is a **wide-format CSV** with these columns:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `subject_id` | string | Yes | Unique subject identifier |
| `age` | numeric | Yes | Age in years |
| `sex` | string | Yes | M / F |
| `site` | string | Yes | Scanner site identifier |
| `breed` | string | Yes | Species/strain |
| `<ROI_1>` | numeric | — | Brain region / metric columns |
| `<ROI_2>` | numeric | — | ... |
| ... | numeric | — | ... |

Each numeric column that is not a metadata column is treated as a separate response variable and gets its own GPR model.

Example:
```csv
subject_id,age,sex,site,breed,V1,M1_hand,dlPFC,OFC,thalamus
sub-001,3.5,M,site_A,Macaca_mulatta,2.34,1.89,1.65,1.92,2.10
sub-002,8.2,F,site_B,Macaca_fascicularis,3.12,2.45,2.10,2.35,2.67
```

Use `--score_prefixes` to restrict which columns are modeled (e.g., `--score_prefixes thickness` only models columns starting with "thickness").

## Usage

### Training

Train GPR normative models from a merged metric CSV.

```bash
python -m gpr_normative.train \
    --csv <path/to/data.csv> \
    --out_dir <output_directory> \
    [--score_prefixes <prefix1,prefix2>] \
    [--exclude_cols <col1,col2>] \
    [--conf_num age] \
    [--conf_cat sex,site,breed] \
    [--nu 2.5] \
    [--n_restarts 1] \
    [--test_size 0.2] \
    [--min_n 20] \
    [--n_jobs 4] \
    [--random_state 0]
```

**Key options:**
- `--test_size`: Fraction of data held out for evaluation (0 = train on all data, no evaluation)
- `--nu`: Matern smoothness parameter (0.5/1.5/2.5). Higher = smoother trajectories. **Default: 2.5**
- `--n_restarts`: Number of optimizer restarts. More = better fit but slower. **Default: 1**
- `--min_n`: Minimum subjects required to train a model for a given ROI. **Default: 20**

**Outputs:**
```
out_dir/
├── models/          # One .joblib file per score column
├── predictions/     # Training set predictions (mu, std, z, resid per ROI)
├── train_table.csv  # Copy of the input data used
├── trained_score_cols.txt
└── test_evaluation.csv  # Only if --test_size > 0
```

### Cross-Validation

Evaluate model performance using K-fold cross-validation (stratified by site to prevent leakage).

```bash
python -m gpr_normative.evaluate \
    --csv <path/to/data.csv> \
    --out_dir <output_directory> \
    [--n_folds 5] \
    [--n_jobs 4]
```

**Outputs `cv_results.csv`** with per-ROI metrics:
| column | description |
|--------|-------------|
| `score_col` | Brain region name |
| `n_samples` | Total subjects |
| `mean_expv` / `std_expv` | Explained variance (higher = better) |
| `mean_mae` / `std_mae` | Mean absolute error |
| `mean_rmse` / `std_rmse` | Root mean squared error |
| `mean_r2` / `std_r2` | R² (coefficient of determination) |

### Prediction

Apply trained models to new subjects to compute deviation scores.

The `predict.py` module **auto-detects** model type:
- **BLR** (pcntoolkit): directory containing `model/normative_model.json`
- **GPR** (sklearn): directory containing `*.joblib` files

```bash
# Predict with pretrained BLR models (from MacaqueGPR project)
python -m gpr_normative.predict \
    --in_path new_subjects.csv \
    --model_dir path/to/M129/L/thickness \
    --out_dir results/predictions

# Predict with GPR models (from quick start training)
python -m gpr_normative.predict \
    --in_path new_subjects.csv \
    --model_dir results/quickstart/models \
    --out_dir results/predictions
```

> **Note**: BLR prediction requires `pcntoolkit` (Python 3.10-3.12): `pip install pcntoolkit`

**Outputs (BLR):**
```
out_dir/
├── predictions.csv                  # Long format: subject_id, roi, predicted_mu, z_score, residual
├── predictions_wide.csv             # Wide format: one row per subject, columns per ROI
└── predict_summary.csv              # Per-ROI status
```

**Outputs (GPR):**
```
out_dir/
├── predictions/
│   ├── <ROI_1>__predictions.csv   # mu, std, resid, z per subject
│   ├── <ROI_2>__predictions.csv
│   └── ...
├── predict_summary.csv            # Summary of all predictions
└── predictions_merged_wide.csv    # All ROIs merged by subject_id
```

The z-score is the key metric: **|z| > 1.96** indicates a statistically significant deviation from the normative range (p < 0.05, two-tailed).

### Pretrained Models

The 2,942 normative models from the MacaqueGPR project are available on Hugging Face Hub.

```bash
# Install downloader
pip install huggingface_hub

# Download all models (~7.6 GB)
python -m gpr_normative.model_hub \
    --repo_id yahuiwei123/MacaqueGPR-models \
    --local_dir ./blr_models
```

Or from Python:
```python
from gpr_normative.model_hub import download_models
path = download_models("yahuiwei123/MacaqueGPR-models", "./models")
```

Model directory structure after download:
```
blr_models/
├── M129/L/{thickness,cortvol,curvature,sulc,area}/   # 80 ROIs each
├── M132/...                                           # 92 ROIs each
├── MBNA124/...                                        # 124 ROIs each
├── Modalities/...                                     # 9-18 ROIs each
├── subcort/...                                        # 15 structures each
└── optimal_parameters_summary.csv
```

### Fine-Tuning

Adapt pretrained models to a new dataset (different population, scanner, or species).

```bash
python -m gpr_normative.fine_tune \
    --csv <custom_data.csv> \
    --model_dir <pretrained/models> \
    --out_dir <finetuned/output> \
    [--freeze_length_scale] \
    [--n_restarts 5] \
    [--n_jobs 4]
```

See the [Fine-Tuning Guide](#fine-tuning-guide) below for detailed strategies.

## Understanding GPR Hyperparameters

The model uses a composite kernel: `ConstantKernel * Matern + WhiteKernel`.

### Kernel Components

```
K_total = σ² · k_matern(ν, ℓ) + σ_n²
```

### Matern Smoothness (ν / nu)

Controls how smooth the fitted normative trajectory is.

| ν | Behavior | When to use |
|----|----------|-------------|
| 0.5 | Rough, non-differentiable (exponential kernel) | Noisy data with sharp transitions |
| 1.5 | Once differentiable, moderately smooth | Most neuroimaging data |
| **2.5** | Twice differentiable, very smooth **(default)** | Smooth developmental trajectories |

Higher ν = smoother curves, which is generally appropriate for brain development where changes are gradual.

### Length Scale (ℓ)

How quickly the function changes with age. A small ℓ allows rapid age-related changes; a large ℓ forces the trajectory to be nearly linear.

After training, you can inspect the learned length scale:
```python
from joblib import load
pipe = load('models/V1.joblib')
kernel = pipe.named_steps['gpr'].kernel_
print(f"constant_value: {kernel.k1.k1.constant_value:.3f}")
print(f"length_scale:   {kernel.k1.k2.length_scale:.3f}")
print(f"noise_level:    {kernel.k2.noise_level:.3f}")
```

### Noise Level (σ_n²)

Captures measurement noise and biological variance not explained by age. A higher noise level means wider normative centiles.

### Constant Value (σ²)

Overall signal variance — the magnitude of variation in the brain measure across the population.

## Fine-Tuning Guide

When you have a new dataset (different species, scanner, or population), you can fine-tune existing models rather than training from scratch. This is especially valuable when your dataset is small.

### Strategy by Dataset Size

#### Small dataset (n < 50 subjects)

Freeze the length scale — it captures the fundamental age-trajectory shape, which is often conserved across populations.

```bash
python -m gpr_normative.fine_tune \
    --csv small_dataset.csv \
    --model_dir pretrained/models \
    --out_dir finetuned \
    --freeze_length_scale \
    --n_restarts 5
```

This only refits `constant_value` and `noise_level`, which adapt to different variance scales.

#### Medium dataset (50 ≤ n < 200)

Refit all hyperparameters with more optimizer restarts:

```bash
python -m gpr_normative.fine_tune \
    --csv medium_dataset.csv \
    --model_dir pretrained/models \
    --out_dir finetuned \
    --n_restarts 10
```

#### Large dataset (n ≥ 200)

Consider training from scratch with `train.py`:

```bash
python -m gpr_normative.train \
    --csv large_dataset.csv \
    --out_dir trained_from_scratch \
    --n_restarts 5 \
    --test_size 0.2
```

### Assessing Fine-Tuning Quality

After fine-tuning, compare CV metrics against the original model:

```bash
# Evaluate fine-tuned model
python -m gpr_normative.evaluate \
    --csv custom_data.csv \
    --out_dir finetuned_cv \
    --n_folds 5
```

Look for:
- **EXPV > 0**: Model explains variance beyond the mean baseline
- **Higher R²**: Better fit to your data's age trends
- **z-score distribution**: Should be approximately N(0,1)

### Programmatic Fine-Tuning

```python
from gpr_normative.fine_tune import fine_tune_one_score, rebuild_kernel
from joblib import load

# Load a pretrained model
pipe = load('models/V1.joblib')
old_gpr = pipe.named_steps['gpr']

# Inspect learned parameters
print(f"length_scale: {old_gpr.kernel_.k1.k2.length_scale:.3f}")
print(f"noise_level:  {old_gpr.kernel_.k2.noise_level:.3f}")
print(f"LML (log marginal likelihood): {old_gpr.log_marginal_likelihood_value_:.3f}")

# Fine-tune
result = fine_tune_one_score(
    df=custom_dataframe,
    score_col='V1',
    model_path='models/V1.joblib',
    out_dir='finetuned',
    conf_num=['age'],
    conf_cat=['sex', 'site', 'breed'],
    freeze_length_scale=True,
    n_restarts=10,
    random_state=42,
)
print(f"Fine-tuned EXPV: {result['expv']:.4f}")
```

### Advanced: Custom Confounds

If your dataset has different confound columns, specify them:

```bash
python -m gpr_normative.train \
    --csv my_data.csv \
    --out_dir results \
    --conf_num age,weight_kg \
    --conf_cat sex,site,breed,scanner_model
```

## API Reference

### Module Structure

```
gpr_normative/
├── data_utils.py    # CSV loading, column detection, data preparation
├── train.py         # GPR training with optional holdout evaluation
├── predict.py       # Apply trained models to compute deviation scores
├── evaluate.py      # K-fold cross-validation with multiple metrics
└── fine_tune.py     # Fine-tune pretrained models on custom data
```

### Key Functions

**`train.py`**
- `build_gpr_pipeline(numeric_cols, categorical_cols, nu, n_restarts_optimizer, random_state)` — Build the GPR sklearn Pipeline
- `train_one_score(df, score_col, conf_num, conf_cat, out_dir, ...)` — Train a single ROI model
- `compute_metrics(y_true, y_pred)` — Compute EXPV, MAE, RMSE, R²

**`predict.py`**
- `predict_one_metric(df, model_path, out_dir, conf_num, conf_cat, id_cols)` — Predict for one metric
- `list_model_files(model_dir)` — List all .joblib models in a directory

**`evaluate.py`**
- `cross_validate_one_score(df, score_col, ..., n_folds)` — K-fold CV for one ROI

**`fine_tune.py`**
- `fine_tune_one_score(df, score_col, model_path, out_dir, ..., freeze_length_scale)` — Fine-tune one model
- `rebuild_kernel(base_kernel, nu, freeze_length_scale)` — Rebuild kernel for fine-tuning

**`data_utils.py`**
- `load_one_merged_metric_csv(csv_path)` — Load and standardize a training CSV
- `load_and_standardize_one(csv_path)` — Load and standardize a prediction CSV
- `find_score_columns(df, prefixes, exclude_cols)` — Identify response variable columns
- `prepare_train_data(df, score_col, conf_num, conf_cat, min_n)` — Prepare X, y for training
- `fill_missing_confounds(df)` — Impute missing confound values

## Citation

If you use this toolkit in your research, please cite the relevant scikit-learn Gaussian Process documentation:
> Wei, Y. et al. MacaSurfer: unified surface-volume mapping of the macaque brain across the lifespan. 2026.06.14.732101 Preprint at https://doi.org/10.64898/2026.06.14.732101 (2026).

> Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

> Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
