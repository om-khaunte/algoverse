# CD-HIT 40% Cross-Species Homology Filter

## Purpose
Prevent homologous proteins from occurring in different species cohorts used
for zero-shot evaluation.

## Input
- Combined positive and negative sequences: 7373
- Species before filtering: 73
- Positive proteins before filtering: 4759
- Negative proteins before filtering: 2614

## CD-HIT configuration
- Identity threshold: 40%
- Word length: 2
- Global identity: enabled
- Accurate clustering mode: enabled
- CPU threads: all available
- Input FASTA: `raw/cdhit40/vf_all_sequences.faa`

## Cross-Species Exclusion Policy
Every sequence in every CD-HIT cluster containing proteins from more than one
species was removed. No representative sequence was retained from a
mixed-species cluster. This symmetric policy prevents arbitrary species
selection and prevents cross-species cluster membership in the retained cohort.

## Results
- Total CD-HIT clusters: 5856
- Cross-species clusters removed: 657
- Sequences before filtering: 7373
- Sequences excluded: 1798
- Sequences retained: 5575
- Species retained: 72
- Positive proteins retained: 3384
- Negative proteins retained: 2191

## Validation
After filtering, each retained CD-HIT cluster contains exactly one species.

## Output Files
- `data/vf_dataset_before_cdhit40.csv`
- `data/vf_dataset_cdhit40_cross_species_filtered.csv`
- `data/vf_cdhit40_cross_species_excluded.csv`
- `data/vf_cdhit40_cluster_summary.csv`
- `data/vf_cdhit40_species_impact.csv`
- `docs/cdhit40_cross_species_filter.md`

## Final Validation
- Retained clusters with multiple species: 0
- Retained positive sequences: 3384
- Retained negative sequences: 2191
- Species zero-shot ready: 42
