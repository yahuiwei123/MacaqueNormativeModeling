#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_DIR="${SAVE_DIR:-/home/weiyahui/projects/Monkey_Surface/experiments/statistic/scripts/postprocess/resources/blr/save_dir}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results}"
INPUT_CSV="${SCRIPT_DIR}/merged_stats/cort/Modalities/R/thickness.csv"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-pcntoolkit}"

mkdir -p "${OUT_DIR}"

echo "Input CSV: ${INPUT_CSV}"
echo "Model save_dir: ${SAVE_DIR}"
echo "Output directory: ${OUT_DIR}"

echo
echo "[1/4] Export expected input template"
"${PYTHON_BIN}" -m monkey_normative.cli template \
  --save-dir "${SAVE_DIR}" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --out-csv "${OUT_DIR}/Modalities_R_thickness_template.csv"

echo
echo "[2/4] Predict deviations using pretrained full_data_model"
"${PYTHON_BIN}" -m monkey_normative.cli predict \
  --save-dir "${SAVE_DIR}" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --csv "${INPUT_CSV}" \
  --out-dir "${OUT_DIR}/pretrained_prediction" \
  --rois AUDITORY,VISION

echo
echo "[3/4] Fine-tune pretrained full_data_model on the OHSU example"
"${PYTHON_BIN}" -m monkey_normative.cli fine-tune \
  --save-dir "${SAVE_DIR}" \
  --atlas Modalities \
  --hemi R \
  --metric thickness \
  --csv "${INPUT_CSV}" \
  --out-dir "${OUT_DIR}/finetuned/Modalities_R_thickness" \
  --min-n 10

echo
echo "[4/4] Predict deviations using the fine-tuned model"
"${PYTHON_BIN}" -m monkey_normative.cli predict \
  --model-dir "${OUT_DIR}/finetuned/Modalities_R_thickness" \
  --csv "${INPUT_CSV}" \
  --out-dir "${OUT_DIR}/finetuned_prediction" \
  --rois AUDITORY,VISION

echo
echo "Done. Key outputs:"
echo "  ${OUT_DIR}/Modalities_R_thickness_template.csv"
echo "  ${OUT_DIR}/pretrained_prediction/predictions_long.csv"
echo "  ${OUT_DIR}/pretrained_prediction/predictions_wide.csv"
echo "  ${OUT_DIR}/finetuned/Modalities_R_thickness/fine_tune_summary.json"
echo "  ${OUT_DIR}/finetuned/Modalities_R_thickness/fine_tune_metrics_by_roi.csv"
echo "  ${OUT_DIR}/finetuned_prediction/predictions_long.csv"
echo "  ${OUT_DIR}/finetuned_prediction/predictions_wide.csv"
