# Data Format

## Expected CSV Format

The GPR normative model expects **wide-format CSV files** with the following structure:

### Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `subject_id` | string | Unique subject identifier |
| `age` | numeric | Age in years (required confound) |
| `sex` | string | M / F (categorical confound) |
| `site` | string | Site/scanner identifier (categorical confound) |
| `breed` | string | Species/breed (categorical confound) |

### Score Columns

All remaining numeric columns are treated as response variables (brain regions/ROIs). Each column gets its own GPR model.

You can restrict which columns are treated as scores using `--score_prefixes` (e.g., `--score_prefixes thickness,area` to only model columns starting with those names).

### Example

```csv
subject_id,age,sex,site,breed,V1,M1_hand,S1,dlPFC,OFC
sub-001,3.5,M,site_A,Macaca_mulatta,2.34,1.89,2.10,1.65,1.92
sub-002,8.2,F,site_B,Macaca_fascicularis,3.12,2.45,2.67,2.10,2.35
```

## Generating Example Data

```bash
python data/generate_example_data.py --out_dir data --n_subjects 200
```
