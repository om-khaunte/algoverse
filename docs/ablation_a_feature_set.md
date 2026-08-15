# Ablation A: AAC + DPC Feature Set

## Purpose
This is the non-language-model baseline for Ablation A. It replaces ESM-2
embeddings with deterministic sequence-composition features while retaining
the exact same canonical dataset rows and `sample_id` identifiers.

## Canonical Input
- `data/vf_dataset_cdhit40_zero_shot_ready.csv`
- Rows: 4551
- Feature dimensions: 420
  - Amino-acid composition (AAC): 20
  - Dipeptide composition (DPC): 400

## Output Files
- `data/vf_ablation_a_aac_dpc.csv`: numeric matrix only, shape `(4551, 420)`.
- `data/vf_ablation_a_aac_dpc_metadata.csv`: `row_index`, `sample_id`, `species`, and `label`.
- `data/vf_ablation_a_feature_columns.json`: immutable feature names/order and normalization metadata.

## Alignment Rule
The feature-matrix row at `row_index = i` corresponds exactly to the metadata
row with `row_index = i`; `sample_id` is the stable join key. Training and
evaluation code must use the existing frozen split sample IDs, then select
matching rows through `data/vf_ablation_a_aac_dpc_metadata.csv`. Do not regenerate splits.

## Feature Definitions
For a canonical 20-amino-acid alphabet
`A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y`:

- `aac_X = count(X) / L`
- `dpc_XY = count(overlapping XY) / (L - 1)`

Ambiguous or noncanonical residue characters are excluded before calculating
features. All retained sequences contain at least two canonical residues.
