"""Smoke tests for GPR Normative Model pipeline."""

import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Add project root to path so we can import gpr_normative
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpr_normative.train import build_gpr_pipeline, compute_metrics, train_one_score
from gpr_normative.predict import predict_one_metric, list_model_files
from gpr_normative.evaluate import cross_validate_one_score
from gpr_normative.data_utils import (
    load_one_merged_metric_csv,
    find_score_columns,
    prepare_train_data,
    fill_missing_confounds,
    lower_and_strip_columns,
    ensure_columns,
    coerce_numeric,
)


def make_sample_df(n=50):
    """Create a minimal test dataframe."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "subject_id": [f"sub-{i:02d}" for i in range(n)],
        "age": rng.uniform(2, 30, n),
        "sex": rng.choice(["M", "F"], n),
        "site": rng.choice(["site_A", "site_B"], n),
        "breed": rng.choice(["B1", "B2"], n),
        "ROI_A": rng.normal(2, 0.5, n),
        "ROI_B": 1.5 + 0.03 * rng.uniform(2, 30, n) + rng.normal(0, 0.2, n),
        "ROI_C": rng.normal(3, 0.3, n),
    })
    # Add realistic age effect to ROI_B
    return df


class TestDataUtils:
    def test_lower_and_strip(self):
        df = pd.DataFrame({"Subject_ID": [1], " Age ": [2]})
        df = lower_and_strip_columns(df)
        assert list(df.columns) == ["subject_id", "age"]

    def test_ensure_columns(self):
        df = pd.DataFrame({"age": [5]})
        df = ensure_columns(df, ["sex", "site"])
        assert "sex" in df.columns
        assert "site" in df.columns

    def test_coerce_numeric(self):
        df = pd.DataFrame({"age": ["5", "N/A", "10"]})
        df = coerce_numeric(df, ["age"])
        assert df["age"].dtype.kind == "f"

    def test_find_score_columns(self):
        df = make_sample_df(10)
        scores = find_score_columns(df, prefixes=("ROI",))
        assert all(c.startswith("ROI") for c in scores)
        assert len(scores) == 3

    def test_find_score_columns_all(self):
        df = make_sample_df(10)
        scores = find_score_columns(df)
        assert "ROI_A" in scores
        assert "ROI_B" in scores
        assert "ROI_C" in scores
        assert "subject_id" not in scores
        assert "age" not in scores

    def test_prepare_train_data(self):
        df = make_sample_df(50)
        X, y, sub = prepare_train_data(df, "ROI_A", ["age"], ["sex", "site", "breed"])
        assert X is not None
        assert len(X) == len(y)
        assert "age" in X.columns

    def test_fill_missing_confounds(self):
        df = pd.DataFrame({"age": [5, np.nan, 10], "sex": ["M", None, "F"], "subject_id": ["a", "b", "c"]})
        result = fill_missing_confounds(df)
        assert result["age"].notna().all()
        assert result["sex"].notna().all()


class TestGPRPipeline:
    def test_build_pipeline(self):
        pipe = build_gpr_pipeline(["age"], ["sex", "site"], nu=1.5)
        assert pipe is not None
        assert hasattr(pipe, "predict")

    def test_train_predict_cycle(self):
        df = make_sample_df(60)
        conf_num = ["age"]
        conf_cat = ["sex", "site", "breed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "test_output"
            result = train_one_score(
                df, "ROI_B", conf_num, conf_cat, out_dir,
                nu=1.5, n_restarts_optimizer=3, random_state=42, min_n=5,
            )
            assert result is not None
            assert result["score_col"] == "ROI_B"

            # Check model file exists
            models = list_model_files(out_dir / "models")
            assert len(models) == 1

            # Predict with same data
            summary = predict_one_metric(
                df, models[0], out_dir,
                conf_num, conf_cat, ["subject_id"],
            )
            assert summary["status"] == "ok"
            assert summary["n_pred"] > 0

    def test_compute_metrics(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        m = compute_metrics(y_true, y_pred)
        assert 0 < m["expv"] < 1
        assert m["mae"] > 0
        assert m["r2"] > 0.9


class TestCrossValidation:
    def test_cv_basic(self):
        df = make_sample_df(60)
        result = cross_validate_one_score(
            df, "ROI_A", ["age"], ["sex", "site", "breed"],
            nu=1.5, n_restarts=2, n_folds=3, random_state=42, min_n=5,
        )
        assert result is not None
        assert "mean_expv" in result
        assert "mean_mae" in result
        assert result["n_folds_completed"] >= 2


class TestEndToEnd:
    def test_full_pipeline(self):
        """Generate data -> train -> evaluate -> predict"""
        df = make_sample_df(80)
        conf_num = ["age"]
        conf_cat = ["sex", "site", "breed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)

            # Save test CSV
            csv_path = base / "test_data.csv"
            df.to_csv(csv_path, index=False)

            # Load it
            loaded = load_one_merged_metric_csv(csv_path)
            assert len(loaded) == 80

            # Train all ROI columns
            out_dir = base / "train_output"
            for sc in ["ROI_A", "ROI_B", "ROI_C"]:
                r = train_one_score(
                    df, sc, conf_num, conf_cat, out_dir,
                    nu=1.5, n_restarts_optimizer=2, random_state=0, min_n=5,
                )
                assert r is not None

            # CV evaluation
            for sc in ["ROI_A", "ROI_B"]:
                r = cross_validate_one_score(
                    df, sc, conf_num, conf_cat,
                    nu=1.5, n_restarts=2, n_folds=3, random_state=0, min_n=5,
                )
                assert r is not None

            # Prediction
            model_dir = out_dir / "models"
            assert len(list_model_files(model_dir)) == 3
