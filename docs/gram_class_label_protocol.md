# Locked Gram-Class Labels for Ablation B

## Purpose
Ablation B tests whether transfer across the Gram-positive/Gram-negative
boundary affects zero-shot performance beyond phylogenetic distance.

The canonical species lookup is:

`data/species_gram_class.csv`

## Scope
The table covers all 42 species in:

`data/vf_dataset_cdhit40_zero_shot_ready.csv`

## Evidence Policy
Labels were manually cross-checked against standard microbiology references and
peer-reviewed organism descriptions. They are not inherited mechanically from
a VFDB, NCBI, GTDB, or BV-BRC metadata field.

## Class Definitions
- `positive`: conventional Gram-positive organism.
- `negative`: conventional Gram-negative organism.
- `acid_fast`: organism with acid-fast biology and unreliable/atypical
  conventional Gram staining.

## Mycobacterium tuberculosis
*Mycobacterium tuberculosis* is explicitly classified as `acid_fast`, not
silently forced into either binary Gram class. It is excluded from the primary
strict Gram-positive-vs-Gram-negative Ablation B comparison. A documented
sensitivity analysis may assign it to the Gram-positive lineage group, but
that is not the primary analysis.

## Lock Rule
The `label_status` field is `locked`. No experiment may infer Gram labels
at runtime from taxonomy or database fields. Any change requires a documented
literature-based correction and a new commit.

## Experimental Usage
For primary strict Ablation B analysis:

```python
gram = pd.read_csv("data/species_gram_class.csv")
strict_gram = gram.query("gram_class in ['positive', 'negative']")
dataset = dataset.merge(
    strict_gram[["species", "gram_class", "gram_binary"]],
    on="species",
    how="inner",
    validate="many_to_one",
)
```
