# Ablation C high-budget expansion status

## Result

The existing `splits/pool_sizes.json` is a frozen final split file and cannot be safely expanded to an `n≈1000` regime by editing counters alone.

## Why

The file already materializes exact `train_sample_ids` for each target and each of the 1-, 3-, and 5-source pools.
The canonical metadata states that all Ablation C runs must consume exactly these target, source, and training sample IDs.
It also states that source species and individual samples are selected deterministically from frozen LOSO species/sample IDs using SHA-256 ordering.

Because the final IDs are already frozen in the file, increasing:
- `common_total_training_budget`
- `n_samples_per_source`
- `total_training_samples`

without regenerating the corresponding `train_sample_ids` would create an invalid split definition.

## Verified current budget range

The currently frozen budgets in the canonical file remain in the low-budget regime (e.g. 120 or 135 depending on target), not near 1000.

## Correct next step

To create a true high-budget Ablation C expansion, regenerate a new companion split file from the upstream deterministic selection logic:

1. Start from the same frozen LOSO species/sample universe.
2. Re-run the deterministic SHA-256 source/sample selection procedure.
3. Produce a new file (recommended name: `splits/pool_sizes_n1000.json`).
4. Preserve the balance rule: identical total rows across 1-, 3-, and 5-source pools, with a common budget divisible by 15.
5. Commit the new generated split file alongside this note.

## Recommendation

Do not overwrite the existing canonical `splits/pool_sizes.json`.
Keep it unchanged and generate a new companion high-budget split file instead.
