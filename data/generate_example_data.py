#!/usr/bin/env python3
"""
Generate synthetic neuroimaging-like data for demonstrating GPR Normative Model.

Creates realistic data with:
- 200 healthy subjects across 5 sites, 3 breeds, 2 sexes
- Age range 2-30 years with nonlinear developmental curves per ROI
- 20 synthetic brain regions with age-dependent trajectories + noise
- Output: wide-format CSV ready for train.py
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_roi_trajectory(age, roi_type="cortical"):
    """
    Generate a realistic developmental trajectory.
    Different ROI types have different patterns.
    """
    age_norm = age / 30.0  # normalize to [0,1]

    if roi_type == "early_peak":
        # Peaks early (e.g., sensory/motor cortices)
        return 2.0 + 1.5 * np.exp(-((age - 4) ** 2) / 20) + 0.3 * age_norm
    elif roi_type == "late_peak":
        # Peaks late (e.g., association cortices)
        return 1.5 + 2.0 * np.exp(-((age - 12) ** 2) / 60) + 0.5 * age_norm
    elif roi_type == "linear_decline":
        # Linear decline (e.g., some GM measures)
        return 3.0 - 0.8 * age_norm
    elif roi_type == "u_shaped":
        # U-shaped trajectory
        return 2.0 + 0.5 * ((age_norm - 0.4) ** 2) * 5
    else:
        # Default: sigmoid-like growth
        return 2.0 + 1.5 / (1 + np.exp(-(age - 8) / 4))


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic brain normative data.")
    ap.add_argument("--out_dir", type=str, default=".",
                    help="Output directory for the generated CSV.")
    ap.add_argument("--n_subjects", type=int, default=200,
                    help="Number of synthetic subjects.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n = args.n_subjects

    # Demographics
    sites = ["site_A", "site_B", "site_C", "site_D", "site_E"]
    breeds = ["Macaca_fascicularis", "Macaca_mulatta"]
    sexes = ["M", "F"]

    site = rng.choice(sites, n)
    breed = rng.choice(breeds, n)
    sex = rng.choice(sexes, n)

    # Age distribution: skewed toward younger ages with a few older
    age = rng.gamma(shape=3.0, scale=3.5, size=n) + 1.5
    age = np.clip(age, 2.0, 30.0)

    # Define 20 synthetic brain regions with varying trajectory types
    roi_types = {
        "V1": "early_peak", "M1_hand": "early_peak", "S1": "early_peak",
        "A1": "early_peak", "MT": "early_peak",
        "dlPFC": "late_peak", "vlPFC": "late_peak", "dmPFC": "late_peak",
        "OFC": "late_peak", "ACC": "late_peak", "PCC": "late_peak",
        "TPJ": "late_peak", "IPS": "late_peak",
        "ITG": "linear_decline", "MTG": "linear_decline", "STG": "linear_decline",
        "FG": "u_shaped", "PHG": "u_shaped",
        "insula": "default", "thalamus": "default",
    }

    data = {
        "subject_id": [f"sub-{i+1:04d}" for i in range(n)],
        "age": age,
        "sex": sex,
        "site": site,
        "breed": breed,
        "weight (kg)": rng.normal(5.0, 2.0, n).clip(1.5, 15.0),
    }

    for roi_name, roi_type in roi_types.items():
        base = generate_roi_trajectory(age, roi_type)
        # Add site-specific offset
        site_offset = {s: rng.normal(0, 0.15) for s in sites}
        offset = np.array([site_offset[s] for s in site])
        # Add noise
        noise = rng.normal(0, 0.2, n)
        # Add sex effect
        sex_effect = np.where(np.array(sex) == "F", -0.15, 0.0)
        values = base + offset + noise + sex_effect
        data[roi_name] = np.round(values, 4)

    df = pd.DataFrame(data)

    out_path = out_dir / "example_brain_data.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {n} subjects with {len(roi_types)} ROIs")
    print(f"Age range: {age.min():.1f} - {age.max():.1f} (mean={age.mean():.1f})")
    print(f"Sites: {dict(zip(*np.unique(site, return_counts=True)))}")
    print(f"Breeds: {dict(zip(*np.unique(breed, return_counts=True)))}")
    print(f"Sexes: {dict(zip(*np.unique(sex, return_counts=True)))}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
