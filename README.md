Source databases: VFDB + BV-BRC.

Positive class: VF proteins.

Negative class: non-VF proteins from the same genomes.

Deduplication: CD-HIT at 40% identity across species.

Split policy: leave-one-species-out, frozen once created.

Minimum species filter: 8–12 species, each with at least 300 labeled proteins.

Required outputs: data/vf_proteins.csv, splits/loso_folds.json, and a species imbalance log.
