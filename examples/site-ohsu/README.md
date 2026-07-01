# OHSU PRIME-DE Fine-Tuning and Prediction Example

This example uses a real `merged_stats` directory copied from:

```text
/home/weiyahui/projects/Monkey_Surface/datasets_v3.0/PRIME-DE/output/site-ohsu/merged_stats
```

The copied example data lives here:

```text
examples/site-ohsu/merged_stats/
├── cort/{MBNA124,Modalities,M129,M132}/{L,R}/{thickness,area,cortvol,curvature,sulc}.csv
└── subcort/aseg/{L,R}/volume.csv
```

These CSV files are already in the wide format expected by `monkey-norm`:

- metadata: `subject_id`, `session_id`, `age`, `sex`, `site`, `breed`
- global metrics such as `global_mean_thickness`
- ROI columns matching the pretrained model response variables

The runnable example below uses:

```text
examples/site-ohsu/merged_stats/cort/Modalities/R/thickness.csv
```

This file has 16 rows, 2 subjects, one site (`ohsu`), one age value (`5.0`),
and 9 Modalities ROI columns. Because the sample size is 16, the fine-tuning
command uses `--min-n 10`.

## Run the complete example

From the repository root:

```bash
PYTHON_BIN=/home/weiyahui/software/anaconda3/envs/pcnnorm/bin/python \
SAVE_DIR=/home/weiyahui/projects/Monkey_Surface/experiments/statistic/scripts/postprocess/resources/blr/save_dir \
bash examples/site-ohsu/run_finetune_predict.sh
```

If `monkey-norm` is installed in your active environment, this is also enough:

```bash
bash examples/site-ohsu/run_finetune_predict.sh
```

For another machine, point `SAVE_DIR` to the downloaded/pretrained `save_dir`
that contains directories such as `Modalities/R/thickness/full_data_model`.

## Individual commands

Export the exact input template expected by the model:

```bash
monkey-norm template \
  --save-dir "$SAVE_DIR" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --out-csv examples/site-ohsu/results/Modalities_R_thickness_template.csv
```

Predict deviations with the pretrained final model:

```bash
monkey-norm predict \
  --save-dir "$SAVE_DIR" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --csv examples/site-ohsu/merged_stats/cort/Modalities/R/thickness.csv \
  --out-dir examples/site-ohsu/results/pretrained_prediction \
  --rois AUDITORY,VISION
```

Fine-tune the pretrained final model on the OHSU example:

```bash
monkey-norm fine-tune \
  --save-dir "$SAVE_DIR" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --csv examples/site-ohsu/merged_stats/cort/Modalities/R/thickness.csv \
  --out-dir examples/site-ohsu/results/finetuned/Modalities_R_thickness \
  --min-n 10
```

Predict deviations with the fine-tuned model:

```bash
monkey-norm predict \
  --model-dir examples/site-ohsu/results/finetuned/Modalities_R_thickness \
  --csv examples/site-ohsu/merged_stats/cort/Modalities/R/thickness.csv \
  --out-dir examples/site-ohsu/results/finetuned_prediction \
  --rois AUDITORY,VISION
```

## Output files

### `Modalities_R_thickness_template.csv`

Header-only CSV showing the required input columns for the selected model:

```text
subject_id,age,sex,breed,AUDITORY,COGNITIVE,EMOTION,GUSTATORY,MOTOR,OLFACTORY,SOMATOSENSORY,VISION,VISION_V1
```

The OHSU file also contains `session_id`, `site`, global metrics, and other
metadata. Extra columns are allowed.

### `pretrained_prediction/`

Created by `monkey-norm predict` with the pretrained `full_data_model`.

- `predictions_long.csv`: one row per input row per ROI.
  Columns include `row_id`, `subject_id`, `session_id`, `roi`, `observed`,
  `predicted_mu`, `z_score`, `residual`, and `logp`.
- `predictions_wide.csv`: one row per input row, with columns such as
  `AUDITORY__mu`, `AUDITORY__z`, `AUDITORY__resid`,
  `AUDITORY__observed`, `VISION__mu`, and `VISION__z`.
- `predict_summary.csv`: model path, number of input rows, number of predicted
  ROIs, and missing-response count.

### `finetuned/Modalities_R_thickness/`

Created by `monkey-norm fine-tune`.

- `model/normative_model.json`: fine-tuned pcntoolkit normative model metadata.
- `model/{ROI}/regression_model.json`: one fine-tuned BLR model per ROI.
- `fine_tune_summary.json`: source model path, input CSV path, sample count,
  number of ROIs, B-spline parameters, and scaler settings.
- `fine_tune_metrics_by_roi.csv`: per-ROI in-sample fit metrics.
- `results/Z_finetune.csv`, `results/centiles_finetune.csv`,
  `results/logp_finetune.csv`, `results/statistics_finetune.csv`: pcntoolkit
  diagnostic outputs generated during fine-tuning.

The OHSU example has a constant `age` column. The toolkit detects this during
fine-tuning and uses `inscaler="none"` for covariates to avoid division by zero
inside pcntoolkit standardization.

### `finetuned_prediction/`

Created by `monkey-norm predict --model-dir <finetuned model>`.

It has the same prediction output structure as `pretrained_prediction/`, but
uses the fine-tuned model instead of the original full-data model.

## Notes

- This is a single-site example, so ComBat harmonization is not meaningful here.
  ComBat requires at least two sites.
- The example restricts prediction to `AUDITORY,VISION` for concise output.
  Remove `--rois AUDITORY,VISION` to predict all 9 Modalities ROIs.
- With only 16 rows, this example demonstrates mechanics, not a recommended
  final scientific fine-tuning sample size.
