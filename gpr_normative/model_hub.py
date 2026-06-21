"""
Download pretrained normative models from Hugging Face Hub.

Supports both BLR (pcntoolkit JSON) and GPR (sklearn joblib) model formats.

Usage:
  # Download all BLR models
  python -m gpr_normative.model_hub --repo_id yahuiwei123/MacaqueGPR-models --local_dir ./blr_models

  # From Python
  from gpr_normative.model_hub import download_models
  path = download_models("yahuiwei123/MacaqueGPR-models", "./models")
"""

import os
from pathlib import Path


def download_models(repo_id: str = "yahuiwei123/MacaqueGPR-models", local_dir: str | None = None):
    """
    Download pretrained normative models from Hugging Face Hub.

    Args:
        repo_id: HuggingFace repository ID.
        local_dir: Where to save models. Default: ./blr_models

    Returns:
        Path to the downloaded model directory.

    Raises:
        ImportError if huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required for model download.\n"
            "Install with: pip install huggingface_hub"
        )

    if local_dir is None:
        local_dir = os.path.join(os.getcwd(), "blr_models")

    print(f"Downloading models from huggingface.co/{repo_id} ...")
    print(f"This is a large download (~7.6 GB of JSON + CSV files). Please be patient.\n")

    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        resume_download=True,
    )

    n_json = len(list(Path(local_path).rglob("*.json")))
    n_csv = len(list(Path(local_path).rglob("*.csv")))
    print(f"Downloaded {n_json} model files + {n_csv} CSV files to: {local_path}")
    print(f"\nUse these models with:")
    print(f"  python -m gpr_normative.predict --model_dir {local_path}/<atlas>/<hemi>/<metric> --csv your_data.csv --out_dir results/")

    return local_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Download pretrained normative models")
    ap.add_argument("--repo_id", type=str, default="yahuiwei123/MacaqueGPR-models")
    ap.add_argument("--local_dir", type=str, default=None)
    args = ap.parse_args()
    download_models(args.repo_id, args.local_dir)
