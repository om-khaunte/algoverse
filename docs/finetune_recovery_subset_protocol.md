# Fine-Tuning Recovery Subsets

## Canonical Files
- `splits/finetune_recovery_subsets.json`
- `splits/finetune_recovery_subsets.csv`
- `splits/finetune_recovery_subset_validation.csv`

## Purpose
These are frozen, stratified target-species adaptation subsets for recovery
curves after zero-shot transfer.

## Design
For every frozen LOSO target species:
- Adaptation fractions: 10%, 25%, and 50% of the target's frozen LOSO test set.
- Independent draws: 3 per fraction.
- Sampling: stratified by binary VF label.
- Training frame: target species test IDs only.
- Acid-fast/Gram status is irrelevant to this sampling procedure.
- Source-training overlap: prohibited and validated as zero.

## Required Recovery-Curve Procedure
For each target species, fraction, and draw:
1. Load exactly the recorded `sample_id` values from the frozen subset file.
2. Fine-tune the pretrained/zero-shot model using only those target examples.
3. Evaluate only on the held-out remainder of that target's original LOSO test set:
   `target_test_ids - selected_adaptation_sample_ids`.
4. Never evaluate on the adaptation examples used for fine-tuning.
5. Record the target species, fraction, draw, seed, adaptation sample count,
   evaluation sample count, and AUROC.
6. Do not regenerate or replace any selected IDs.

## Caveat
The 10%, 25%, and 50% draws are independently sampled rather than nested.
They measure recovery at each labeled-data budget and should not be treated as
within-draw incremental additions.
