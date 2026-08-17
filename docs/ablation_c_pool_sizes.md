# Ablation C: Frozen Source-Pool Definitions

## Purpose
Ablation C tests whether using a more diverse source-species pool improves
zero-shot transfer. Each held-out target species has pre-specified training
pools containing 1, 3, or 5 source species.

## Canonical Split File
`split/pool_sizes.json`

## Inputs
- `data/vf_dataset_cdhit40_zero_shot_ready.csv`
- `splits/loso_folds.json`

## Balance Rule
For each target, the total training-sample budget is identical across its
1-, 3-, and 5-species source pools. The common budget is divisible by 15,
allowing equal allocation across 1, 3, and 5 source species.

Thus, pool size changes source diversity, not total number of training rows.

## Source and Sample Selection
Source species and individual samples are selected by deterministic SHA-256
ordering from the already frozen LOSO species and `sample_id` values, using
the recorded seed. Selection never uses labels, model results, or
phylogenetic-distance outcomes.

## Freeze Rule
`splits/pool_sizes.json` is canonical. Do not regenerate it. All Ablation C
runs must consume exactly these target, source, and training sample IDs.
