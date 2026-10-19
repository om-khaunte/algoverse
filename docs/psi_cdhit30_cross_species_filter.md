# PSI-CD-HIT 30% Cross-Species Homology Sensitivity Analysis

## Purpose
This analysis is a strict homology-control sensitivity analysis for the primary CD-HIT-40 LOSO evaluation. It does not replace the CD-HIT-40 primary analysis.

## Input
- Combined proteins before homology filtering: 7,373
- Species before homology filtering: 73
- Input dataset: `data/vf_dataset_before_cdhit40.csv`

## Clustering
- Program: PSI-CD-HIT (`psi-cd-hit.pl`) with BLAST+
- Protein sequence-identity threshold: 30%
- Identity definition: global identity
- Clustering mode: accurate (`-g 1`)
- Cross-species rule: exclude every protein in every cluster containing sequences from more than one species.

## Filtering result
- Total clusters: 5,252
- Single-species clusters retained: 4,432
- Mixed-species clusters removed: 820
- Proteins excluded: 2,543
- Proteins retained: 4,830
- Retained species: 68
- Retained negatives: 1,969
- Retained positives: 2,861
- Retained mixed-species clusters: 0 (validation pass)

## LOSO cohort
The predefined eligibility rule requires both labels and at least 300 retained proteins per species. Only two species remain eligible after the strict 30% filter:

- Legionella pneumophila: 703 proteins (344 negative, 359 positive)
- Pseudomonas aeruginosa: 433 proteins (270 negative, 163 positive)

Therefore, the planned 12-target LOSO sensitivity rerun is not feasible under the original eligibility rule. The available dataset supports only two directional species-disjoint evaluations:

1. Train on Pseudomonas aeruginosa (433), test on Legionella pneumophila (703).
2. Train on Legionella pneumophila (703), test on Pseudomonas aeruginosa (433).

## Interpretation constraint
Report these results as a reduced-cohort robustness check. Do not describe them as a direct 12-target comparison with the primary CD-HIT-40 LOSO evaluation.

## Handoff files
- Filtered full dataset: `data/vf_dataset_psi_cdhit30_cross_species_filtered.csv`
- Reduced zero-shot dataset: `data/vf_dataset_psi_cdhit30_zero_shot_ready.csv`
- LOSO folds: `splits/loso_folds_psi_cdhit30_reduced_cohort.json`
- Cluster membership summary: `data/vf_psi_cdhit30_cluster_summary.csv`
- Excluded proteins: `data/vf_psi_cdhit30_cross_species_excluded.csv`
- Species impact table: `data/vf_psi_cdhit30_species_impact.csv`
- Cohort manifest: `data/vf_psi_cdhit30_zero_shot_cohort_manifest.csv`
