#!/usr/bin/env python3
"""
Upload trained BLR normative models to Hugging Face Hub.

Prerequisites:
  1. pip install huggingface_hub
  2. huggingface-cli login

The models are pcntoolkit BLR (Bayesian Linear Regression) normative models stored as JSON.

Model directory structure (from blr/save_dir):
  save_dir/
  ├── M129/L/thickness/model/...     # 80 ROIs per metric
  ├── M129/L/cortvol/model/...
  ├── ...
  ├── M132/...
  ├── MBNA124/...
  ├── Modalities/...
  ├── subcort/...
  └── optimal_parameters_summary.csv

Total: ~7,468 individual ROI models across all atlases, hemispheres, metrics.

Usage:
  python scripts/upload_models_to_hf.py --model_dir /path/to/blr/save_dir --repo_id yahuiwei123/MacaqueGPR-models
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Upload BLR normative models to Hugging Face Hub")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="Path to the blr/save_dir containing trained models.")
    ap.add_argument("--repo_id", type=str, default="yahuiwei123/MacaqueGPR-models",
                    help="HuggingFace repo ID.")
    ap.add_argument("--private", action="store_true",
                    help="Create a private repo.")
    ap.add_argument("--dry_run", action="store_true",
                    help="List files without uploading.")
    args = ap.parse_args()

    from huggingface_hub import HfApi, create_repo, upload_folder

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"Model directory not found: {model_dir}")

    # Count files
    json_files = sorted(model_dir.rglob("*.json"))
    csv_files = sorted(model_dir.rglob("*.csv"))
    total_size = sum(f.stat().st_size for f in json_files + csv_files)

    print(f"BLR model directory: {model_dir}")
    print(f"  JSON model files: {len(json_files)}")
    print(f"  CSV files: {len(csv_files)}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")

    if args.dry_run:
        print("\nFirst 10 files to upload:")
        for f in (json_files + csv_files)[:10]:
            print(f"  {f.relative_to(model_dir)}")
        print(f"  ... and {len(json_files) + len(csv_files) - 10} more")
        return

    print(f"\nCreating repo: {args.repo_id}")
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
    except Exception as e:
        print(f"  Note: {e}")

    print(f"\nUploading to https://huggingface.co/{args.repo_id}")
    print("This will take a while...\n")

    upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        path_in_repo=".",
        commit_message="Upload BLR normative models (7,468 models, pcntoolkit JSON format)",
    )

    print(f"\nDone! Models available at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
