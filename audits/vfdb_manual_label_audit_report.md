# VFDB Manual Positive-Label Audit

## Scope
- Audit date: 2026-08-17
- Sample design: 3 randomly selected eligible species × 20 positive VF-labeled records
- Total sampled annotations: 60
- Reviewed positive labels supported by VFDB: 60
- Reviewed positive labels unsupported or incorrect: 0
- Uncertain mappings: 0

## Observed Error Rate
Primary observed positive-label error rate, counting uncertain records in the denominator:

\[
0/60 = 0.0000%
\]

Among resolved reviews only, the unsupported-label rate was:

\[
0/60 = 0.0000%
\]

This is a small, non-stratified manual spot-check of positive labels only. It does
not estimate false-negative labels, species-wide rates, or the total dataset error rate.

## Results by Species

| species                  |   n_records |   supported |   unsupported |   uncertain |   observed_error_rate |
|:-------------------------|------------:|------------:|--------------:|------------:|----------------------:|
| Escherichia coli         |          20 |          20 |             0 |           0 |                     0 |
| Legionella pneumophila   |          20 |          20 |             0 |           0 |                     0 |
| Streptococcus agalactiae |          20 |          20 |             0 |           0 |                     0 |

## Unsupported-Label Types

No unsupported labels were identified.

## Limitations-Section Wording

“In a reproducible manual audit of 60 randomly sampled positive VF annotations
(20 records from each of three eligible species), 0 annotations
were judged unsupported or incorrectly mapped against VFDB, corresponding to an
observed positive-label error rate of 0.0%. 0 records
had insufficient identifiers for definitive adjudication. This limited spot-check
does not assess false negatives or quantify label quality across all taxa.”
