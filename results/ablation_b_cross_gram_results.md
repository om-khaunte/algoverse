# Ablation B: Cross-Gram Transfer

## Design
A binary Gram-boundary transfer experiment was run using the locked species-level
Gram labels in `data/species_gram_class.csv`. Records labeled `acid_fast` were
excluded rather than assigned to either binary Gram group.

The model was the AAC+DPC composition MLP used in Ablation A:
- 420 input features: 20 amino-acid composition and 400 dipeptide composition
  features
- Architecture: 420→32→1 sigmoid MLP
- Three random seeds: 0, 1, 2
- Training-only standardization within each transfer direction
- Metric: AUROC

## Observed Cross-Gram Results
- Gram-positive → Gram-negative: mean AUROC 0.6451
  (SD 0.0027; n=3 seeds;
  train=446; test=3820).
- Gram-negative → Gram-positive: mean AUROC 0.7580
  (SD 0.0009; n=3 seeds;
  train=3820; test=446).

## Distance-Matched Comparison
The phylogenetic distance matrix (`data/phylo_distance.csv`) is absent, so no distance-matched prediction or Gram-beyond-distance claim is made.

## Interpretation Constraint
These results quantify transfer across the binary Gram boundary. They do not,
by themselves, demonstrate an effect beyond phylogenetic distance. That requires
a pre-specified distance-matched prediction using the recovered phylogenetic
distance matrix and compatible per-target transfer results.
