# Ablation A: Composition-Feature LOSO Baseline

## Question
Does cross-species transfer performance require protein-language-model
representations, or can raw sequence-composition features achieve comparable
zero-shot performance?

## Features
- 20 normalized amino-acid-composition (AAC) features
- 400 normalized overlapping dipeptide-composition (DPC) features
- 420 total features
- No species identifiers, labels, taxonomy, or metadata were provided as model features.

## Evaluation
- Canonical dataset: `data/vf_dataset_cdhit40_zero_shot_ready.csv`
- Splits: `splits/loso_folds.json`
- 12 frozen leave-one-species-out target folds
- Three random seeds per target fold: 0, 1, 2
- Model: fresh 420→32→1 sigmoid MLP per fold and seed
- Metric: AUROC

## Result
- 36 fold-by-seed runs completed (12 folds × 3 seeds).
- Mean AUROC across all fold-by-seed runs: 0.6662.
- Mean target-level AUROC: 0.6662.

## Interpretation Constraint
This establishes the composition-feature baseline. Any claim about the added
value of ESM-2 must use a matched ESM-2 LOSO result generated from the same
canonical samples, frozen folds, labels, seeds, and AUROC calculation.
