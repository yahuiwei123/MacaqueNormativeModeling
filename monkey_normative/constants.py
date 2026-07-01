from __future__ import annotations

from pathlib import Path


DEFAULT_HEALTHY_DATA_DIR = Path(
    "/home/weiyahui/projects/Monkey_Surface/experiments/statistic/healthy_data"
)
DEFAULT_SAVE_DIR = Path(
    "/home/weiyahui/projects/Monkey_Surface/experiments/statistic/scripts/"
    "postprocess/resources/blr/save_dir"
)

ATLAS_LIST = ("MBNA124", "Modalities", "M129", "M132")
HEMI_LIST = ("L", "R")
CORTICAL_METRICS = ("cortvol", "curvature", "sulc", "thickness", "area")

GLOBAL_VAR_MAPPING = {
    "thickness": "global_mean_thickness",
    "area": "global_total_gmarea",
    "cortvol": "global_total_cortvol",
    "curvature": "global_mean_curvature",
    "sulc": "global_mean_sulc",
}

GLOBAL_METRIC_TO_SOURCE = {
    f"global_{metric}": metric for metric in GLOBAL_VAR_MAPPING
}

SUBCORT_GLOBAL_VARS = (
    "global_estimated_ICV_mm3",
    "global_total_subcortical_vol_mm3",
)

SPLIT_META_COLS = {
    "subject_id",
    "participant_id",
    "age",
    "sex",
    "site",
    "breed",
    "weight (kg)",
    "atlas",
    "hemisphere",
}

FULL_META_COLS = {
    "subject_id",
    "participant_id",
    "age",
    "sex",
    "site",
    "breed",
    "weight (kg)",
    "scan_id",
    "session_id",
    "atlas",
    "hemisphere",
}

COMBAT_META_COLS = FULL_META_COLS | {
    "aseg_path",
    "voxel_volume_mm3",
}
