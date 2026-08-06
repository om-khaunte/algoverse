# Leave-One-Species-Out (LOSO) Fold Definitions

## Source Dataset
- File: `data/vf_dataset_cdhit40_zero_shot_ready.csv`
- Description: CD-HIT 40% cross-species homology-filtered, zero-shot-ready VF dataset.

## LOSO Species Selection
- Minimum sequences per species: 300
- Number of LOSO species: 12
- Frozen species list stored in `splits/loso_species.json`.
- Species were selected once by sorting species by total sequence count and
  taking the top 12 with at least 300 sequences.

## Fold Definition
For each LOSO species, we define one fold:
- Test set: all samples where `species == test_species`.
- Train set: all remaining samples in the zero-shot-ready dataset.
- Folds are stored in `splits/loso_folds.json` and must **never** be regenerated.

These folds are model-agnostic and specified in terms of sample IDs, so they can
be reused across different architectures and training runs.

## Files
- `splits/loso_species.json`: frozen list of LOSO species.
- `splits/loso_folds.json`: complete LOSO fold definitions and counts.
