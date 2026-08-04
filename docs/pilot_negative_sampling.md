# VFDB Pilot Negative Sampling Audit

## Pilot organism
- VFDB organism: Legionella pneumophila subsp. pneumophila str. Philadelphia 1
- Species: Legionella pneumophila
- BV-BRC genome ID: 272624.6
- Reference assembly: GCA_000008485.1 / AE017354.1

## Dataset counts
- VFDB-positive proteins: 461
- Sampled negative proteins: 461
- Positive:negative ratio: 1.0:1

## Negative eligibility
A BV-BRC protein was eligible only if it:
1. Was not annotated in the BV-BRC specialty-gene table with `property = Virulence Factor`.
2. Did not exactly match a VFDB-positive protein sequence from this pilot organism.
3. Was unique by amino-acid sequence.

## Length matching
- Method: nearest_length_without_replacement
- Random seed: 42
- Median absolute positive-negative length difference: 0.0 aa
- Maximum absolute positive-negative length difference: 244 aa
- Kolmogorov-Smirnov statistic: 0.0087
- Kolmogorov-Smirnov p-value: 1

## Output files
- `data/vf_negatives_legionella_pneumophila_philadelphia1.csv`
- `data/vf_negatives.csv`
- `data/vf_species_ratios.csv`

## Scope
This is a reproducible pilot dataset for one exact strain. The aggregate negative dataset must be expanded by applying the same workflow to each mapped organism in `data/bvbrc_genome_manifest.csv`.
