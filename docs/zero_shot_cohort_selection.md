# Zero-Shot Evaluation Cohort Selection

## Source Dataset
The cohort was derived from:

`data/vf_dataset_cdhit40_cross_species_filtered.csv`

The source data have already undergone CD-HIT filtering at 40% sequence
identity, with every sequence from cross-species clusters removed.

## Eligibility Rule
A species is eligible for per-species binary zero-shot evaluation only if it
contains at least one positive and one negative example after homology
filtering.

Species containing a single label class remain in the full filtered dataset but
are excluded from the binary evaluation cohort. Per-species AUROC, AUPRC,
accuracy, F1, precision, and recall are not meaningful for a species with only
one label class.

## Cohort Counts
- Species after CD-HIT filtering: 72
- Eligible zero-shot species: 42
- Ineligible single-class species: 30
- Zero-shot-ready sequences: 4551
- Positive zero-shot-ready sequences: 2364
- Negative zero-shot-ready sequences: 2187

## Output Files
- `data/vf_dataset_cdhit40_zero_shot_ready.csv`
- `data/vf_dataset_cdhit40_not_zero_shot_ready.csv`
- `data/vf_cdhit40_zero_shot_cohort_manifest.csv`
