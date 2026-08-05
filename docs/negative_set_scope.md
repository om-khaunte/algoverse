# Negative-Set Scope and Sampling Policy

## Task
Build the negative set: non-VF proteins from the same genomes, matched on
length distribution. Log the positive:negative ratio per species.

## Final policy
Only VFDB organisms with a verified exact source-genome mapping were included.
Same-species different-strain substitutions were not used.

## Included cohort
- VFDB organism records total: 169
- Exact same-genome organism records completed: 73
- Records excluded because no verified exact source genome was available: 96
- Included positive proteins: 2,614
- Included length-matched negative proteins: 2,614
- Organisms with complete 1:1 matching: 73
- Organisms with insufficient safe candidates: 0

## Negative eligibility
A candidate negative was retained only when it:
1. Came from the exact mapped genome for its VFDB organism.
2. Was not an exact sequence match to any VFDB-positive protein.
3. Had no DIAMOND hit to a VFDB-positive protein at >=30% identity,
   >=70% query coverage, and E-value <=1e-5.

## Sampling
One unused safe negative was selected for each positive using nearest protein
length matching without replacement, with random seed 42.

## Species-ratio interpretation
`sampled_positive_to_negative_ratio` is the ratio in the delivered matched
baseline set. It is 1.0 for fully matched species by design.
`safe_candidate_to_positive_ratio` records the available non-VF candidate
imbalance before 1:1 length-matched sampling and is the ratio relevant for
alternative imbalance-aware baselines.

## Scope limitation
The dataset is a strict exact-genome benchmark cohort, not a same-species
fallback cohort. The 96 excluded records must not be interpreted as negatives
or as evidence that no genome exists; they lack verified source-genome mapping.
