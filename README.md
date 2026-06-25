# Macaque BLR Normative Modeling Toolkit

Bayesian Linear Regression (BLR) toolkit for normative modeling of brain structural MRI data. Trains normative models with B-spline basis functions on healthy subjects, evaluates via cross-validation, and predicts deviation z-scores for new subjects.

This is the training and inference code accompanying the Macaque Normative Atlas (MacNA) project, which constructed **2,942 regional normative models** from a multi-site cohort of 835 neurologically healthy macaques (1,145 imaging sessions across 26 international sites).

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Usage](#usage)
  - [Step 1: ComBat Harmonization](#step-1-combat-harmonization)
  - [Step 2: Training](#step-2-training)
  - [Step 3: Prediction](#step-3-prediction)
  - [Step 4: Fine-Tuning](#step-4-fine-tuning)
- [Pretrained Models](#pretrained-models)
- [Fine-Tuning Guide](#fine-tuning-guide)
- [Module Reference](#module-reference)
- [Citation](#citation)

## Pipeline Overview

```
Raw multi-site data
  │
  ├─ Step 1: ComBat harmonization (combat.py)
  │   Remove site effects, preserve age trends
  │
  ├─ Step 2: BLR training (blr_train.py)
  │   K-fold CV → optimal params → train/test evaluation
  │
  ├─ Step 3: Prediction (predict.py)
  │   Auto-detects BLR vs GPR models → z-scores
  │
  └─ Step 4: Fine-tuning (blr_fine_tune.py)
      Adapt pretrained models to new populations
```

**Model specification**: BLR with B-spline basis (degree=2-3, nknots=2-5), fixed effects for covariates (age + global metrics), batch effects for sex/breed, heteroskedastic noise modeling, sinh-arcsinh warping (optional).

## Installation

```bash
git clone https://github.com/yahuiwei123/MacaqueGPR.git
cd MacaqueGPR
pip install -r requirements.txt
```

**Key dependencies**:

| Package | Purpose | Install |
|---------|---------|---------|
| `pcntoolkit` | BLR training & prediction | `pip install pcntoolkit` (Python 3.10-3.12) |
| `neuroHarmonize` | ComBat harmonization | `pip install neuroHarmonize` |
| `huggingface_hub` | Model download | `pip install huggingface_hub` |
| `scikit-learn` | GPR fallback, data splitting | `pip install scikit-learn` |

> **Note**: `pcntoolkit` requires Python 3.10–3.12. Create a dedicated conda environment if needed:
> ```bash
> conda create -n macaque-gpr python=3.12
> conda activate macaque-gpr
> pip install pcntoolkit neuroHarmonize scikit-learn pandas joblib
> ```

## Quick Start

```bash
# 1. Generate example data (200 subjects, 20 synthetic ROIs)
python data/generate_example_data.py --out_dir data --n_subjects 200

# 2. Train a BLR model with CV optimization
python -m gpr_normative.blr_train \
    --csv data/example_brain_data.csv \
    --out_dir results/quickstart \
    --n_folds 3

# 3. Predict deviations
python -m gpr_normative.predict \
    --in_path data/example_brain_data.csv \
    --model_dir results/quickstart \
    --out_dir results/predictions

# 4. Fine-tune on custom data
python -m gpr_normative.blr_fine_tune \
    --model_dir results/quickstart \
    --csv my_custom_data.csv \
    --out_dir results/finetuned
```

## Data Format

Input CSV files are **wide-format**, one row per subject:

| Column | Type | Required | Description |
|--------|------|----------|-------------|
| `subject_id` | string | Yes | Unique identifier |
| `age` | numeric | Yes | Age in years (primary covariate) |
| `sex` | string | Yes | M / F (batch effect) |
| `breed` | string | Yes | Species/strain (batch effect) |
| `site` | string | For ComBat | Scanner site (harmonized out before training) |
| `global_mean_thickness` | numeric | Optional | Global metric used as additional covariate |
| `<ROI_1>`, `<ROI_2>`, ... | numeric | — | Brain region measurements |

Example:
```csv
subject_id,age,sex,breed,site,global_mean_thickness,10_MMF11,11_MMF11,12_MMF11
sub-001,3.5,M,Macaca_mulatta,site_A,1.52,2.34,1.89,2.10
sub-002,8.2,F,Macaca_fascicularis,site_B,1.48,3.12,2.45,2.67
```

## Usage

### Step 1: ComBat Harmonization

Remove scanner/site effects from multi-site data before training.

```bash
# Harmonize a single file
python -m gpr_normative.combat \
    --csv data/M129/L/thickness.csv \
    --output_dir data/harmonized

# Batch harmonize all CSVs in a directory tree
python -m gpr_normative.combat \
    --input_dir data/ \
    --output_dir data/harmonized
```

ComBat uses `site` as the batch variable and preserves `age` as a smooth biological effect. Files with < 2 sites or zero-variance features are automatically skipped.

### Step 2: Training

Train a BLR normative model with K-fold cross-validation hyperparameter optimization.

```bash
python -m gpr_normative.blr_train \
    --csv data/harmonized/thickness.csv \
    --out_dir results/trained/thickness \
    --covariates age,global_mean_thickness \
    --batch_effects sex,breed \
    --n_folds 5 \
    --test_size 0.2
```

**Hyperparameters optimized via CV:**

| Parameter | Values | Description |
|-----------|--------|-------------|
| `nknots` | 2, 3, 4, 5 | B-spline interior knots (more = more flexible) |
| `degree` | 2, 3 | B-spline polynomial degree |
| `heteroskedastic` | True, False | Model variance as function of age |
| `adaptive_knots` | True, False | Place knots at data quantiles vs uniform |
| `warp_name` | None, "warpsinharcsinh" | Apply sinh-arcsinh warping to response |

**Outputs:**
```
results/trained/thickness/
├── model/                          # pcntoolkit NormativeModel
│   ├── normative_model.json        # Model config, scalers
│   ├── {ROI_NAME}/
│   │   └── regression_model.json   # Per-ROI fitted BLR
│   └── ...
├── cross_validation_results.csv    # All CV param combos ranked by EXPV
└── test_metrics_by_roi.csv         # Per-ROI test set performance
```

### Step 3: Prediction

The `predict.py` module **auto-detects** model type and predicts deviation scores.

```bash
# Predict with trained BLR model
python -m gpr_normative.predict \
    --in_path new_subjects.csv \
    --model_dir results/trained/thickness \
    --out_dir results/predictions
```

**Output** (long format):
```
subject_id, roi, observed, predicted_mu, residual, z_score
sub-001, 10_MMF11, 2.34, 2.41, -0.07, -0.52
sub-001, 11_MMF11, 1.89, 1.95, -0.06, -0.38
```

**Interpretation**: `|z_score| > 1.96` indicates statistically significant deviation from the normative range (p < 0.05, two-tailed).

### Step 4: Fine-Tuning

Adapt a pretrained model to a new dataset (e.g., different species, age range, or scanner).

```bash
python -m gpr_normative.blr_fine_tune \
    --model_dir results/trained/thickness \
    --csv my_new_population.csv \
    --out_dir results/finetuned/thickness
```

**Options:**
- `--nknots 3`: Override B-spline knot count
- `--degree 2`: Override B-spline degree
- `--heteroskedastic true`: Enable/disable heteroskedastic noise

The fine-tuner preserves the model structure (covariates, batch effects) but re-fits all parameters on the new data. B-spline knots are recalculated based on the new data's age distribution.

See the [Fine-Tuning Guide](#fine-tuning-guide) for detailed strategies.

## Pretrained Models

The 2,942 normative models from the MacaqueGPR project are available on Hugging Face Hub.

### Download

```bash
# Install downloader
pip install huggingface_hub

# Download all models
python -m gpr_normative.model_hub \
    --repo_id yahuiwei123/MacaqueGPR-models \
    --local_dir ./pretrained_models
```

Or from Python:
```python
from gpr_normative.model_hub import download_models
path = download_models("yahuiwei123/MacaqueGPR-models", "./models")
```

### Model Inventory

```
pretrained_models/              # blr/save_dir structure
├── M129/                       # M129 atlas
│   ├── L/                      # Left hemisphere
│   │   ├── thickness/model/    # 80 ROI models
│   │   ├── cortvol/model/      # 80 ROI models
│   │   ├── curvature/model/    # 80 ROI models
│   │   ├── sulc/model/         # 80 ROI models
│   │   └── area/model/         # 160 ROI models
│   └── R/                      # Right hemisphere (same structure)
├── M132/                       # M132 atlas (92 ROIs per metric)
├── MBNA124/                    # MacBNA124 atlas (124 ROIs per metric)
├── Modalities/                 # Multi-modal atlas (9-18 ROIs per metric)
├── subcort/                    # Subcortical aseg labels (15 structures)
└── optimal_parameters_summary.csv
```

| Atlas | ROIs per metric | Metrics | Models |
|-------|----------------|---------|--------|
| M129 | 80 | 5 | 800 |
| M132 | 92 | 5 | 920 |
| MBNA124 | 124 | 5 | 1,240 |
| Modalities | 9–18 | 5 | 90 |
| Subcortical | 15 | 1 | 30 |
| Global variables | 1 | 6 | 24 |
| **Total** | | | **3,104** |

## Fine-Tuning Guide

### When to Fine-Tune vs Train from Scratch

| Scenario | Recommendation | Rationale |
|----------|---------------|-----------|
| New species (similar to macaque) | Fine-tune | Age trajectory shape is largely conserved |
| New scanner/site | Fine-tune | Only noise/scale parameters differ |
| Very different age range | Fine-tune with `--nknots` adjustment | More knots for wider age spans |
| Completely different modality | Train from scratch | B-spline basis may not transfer |
| Very small dataset (n < 30) | Fine-tune only | Not enough data for CV optimization |

### Strategy by Dataset Size

**Small (n < 50):** Keep original B-spline structure, just re-fit.
```bash
python -m gpr_normative.blr_fine_tune \
    --model_dir pretrained/M129/L/thickness \
    --csv small_dataset.csv \
    --out_dir finetuned
```

**Medium (50 ≤ n < 200):** Optimize structure with more knots.
```bash
python -m gpr_normative.blr_fine_tune \
    --model_dir pretrained/M129/L/thickness \
    --csv medium_dataset.csv \
    --out_dir finetuned \
    --nknots 4 --heteroskedastic true
```

**Large (n ≥ 200):** Train from scratch with full CV.
```bash
python -m gpr_normative.blr_train \
    --csv large_dataset.csv \
    --out_dir trained_from_scratch \
    --n_folds 5
```

### Evaluating Fine-Tuning Quality

```bash
# Predict on your data and check z-score distribution
python -m gpr_normative.predict \
    --in_path my_data.csv \
    --model_dir finetuned \
    --out_dir eval

# Healthy subjects should have z-scores ~ N(0, 1)
python -c "
import pandas as pd
df = pd.read_csv('eval/predictions_long.csv')
print(f'Mean z: {df[\"z_score\"].mean():.3f}')
print(f'Std z:  {df[\"z_score\"].std():.3f}')
print(f'|z|>1.96: {(df[\"z_score\"].abs() > 1.96).mean()*100:.1f}%')
"
# Expected: Mean z ≈ 0, Std z ≈ 1, ~5% exceed 1.96
```

## Module Reference

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| `combat.py` | Multi-site harmonization | `harmonize_csv()`, `harmonize_directory()` |
| `blr_train.py` | BLR training with CV | `train_blr_model()`, `cross_validation_optimize()` |
| `blr_predict.py` | BLR prediction via pcntoolkit | `predict_blr_models()` |
| `blr_fine_tune.py` | Fine-tune pretrained BLR | `fine_tune_blr()` |
| `predict.py` | Unified prediction (auto-detect BLR/GPR) | `main()` |
| `data_utils.py` | CSV loading & preprocessing | `load_one_merged_metric_csv()`, `find_score_columns()` |
| `model_hub.py` | Download pretrained models | `download_models()` |

**Legacy GPR modules** (`train.py`, `evaluate.py`, `fine_tune.py`): sklearn-based Gaussian Process Regression — simpler, no pcntoolkit dependency, suitable for quick prototyping.

## Citation

If you use this toolkit or the pretrained models, please cite:

> Wei, Y. et al. MacaSurfer: unified surface-volume mapping of the macaque brain across the lifespan. 2026.06.14.732101 Preprint at https://doi.org/10.64898/2026.06.14.732101 (2026).

> Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

> Rasmussen, C.E. & Williams, C.K.I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.

> Fraza, C. et al. (2021). Warped Bayesian linear regression for normative modelling of big data. *NeuroImage*, 245, 118715.
