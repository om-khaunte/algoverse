from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
DIPEPTIDES = [a1 + a2 for a1 in AMINO_ACIDS for a2 in AMINO_ACIDS]
FEATURE_COLUMNS = (
    [f"aac_{aa}" for aa in AMINO_ACIDS]
    + [f"dpc_{dipeptide}" for dipeptide in DIPEPTIDES]
)

INPUT_PATH = Path("data/vf_dataset_cdhit40_zero_shot_ready.csv")
FEATURE_PATH = Path("data/vf_ablation_a_aac_dpc.csv")
METADATA_PATH = Path("data/vf_ablation_a_aac_dpc_metadata.csv")
COLUMNS_PATH = Path("data/vf_ablation_a_feature_columns.json")
DOC_PATH = Path("docs/ablation_a_feature_set.md")


def normalize_sequence(sequence: object) -> str:
    """Uppercase a sequence and retain canonical amino acids only."""
    seq = str(sequence).upper()
    return "".join(aa for aa in seq if aa in AMINO_ACIDS)


def aac_dpc_vector(sequence: object) -> np.ndarray:
    """Return normalized AAC (20) + overlapping DPC (400) features."""
    seq = normalize_sequence(sequence)
    features = np.zeros(420, dtype=np.float32)

    length = len(seq)
    if length > 0:
        for aa in seq:
            features[AMINO_ACIDS.index(aa)] += 1.0
        features[:20] /= length

    if length > 1:
        for i in range(length - 1):
            dipeptide = seq[i:i + 2]
            first = AMINO_ACIDS.index(dipeptide[0])
            second = AMINO_ACIDS.index(dipeptide[1])
            features[20 + first * 20 + second] += 1.0
        features[20:] /= (length - 1)

    return features


def main() -> None:
    assert INPUT_PATH.exists(), f"Missing canonical dataset: {INPUT_PATH}"

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    required = {"sample_id", "species", "sequence", "label"}
    missing = required - set(df.columns)
    assert not missing, f"Input is missing columns: {sorted(missing)}"
    assert df["sample_id"].notna().all(), "sample_id contains missing values."
    assert df["sample_id"].is_unique, "sample_id must be unique."
    assert df["sequence"].notna().all(), "sequence contains missing values."

    vectors = np.vstack([aac_dpc_vector(seq) for seq in df["sequence"]])
    feature_df = pd.DataFrame(vectors, columns=FEATURE_COLUMNS)

    metadata_df = df[
        ["sample_id", "species", "label"]
    ].copy()
    metadata_df.insert(0, "row_index", np.arange(len(df), dtype=int))

    assert len(feature_df) == len(metadata_df) == len(df)
    assert feature_df.shape[1] == 420
    assert np.isfinite(feature_df.to_numpy()).all()
    assert ((feature_df.iloc[:, :20].sum(axis=1) - 1.0).abs() < 1e-6).all()
    assert ((feature_df.iloc[:, 20:].sum(axis=1) - 1.0).abs() < 1e-6).all()

    feature_df.to_csv(FEATURE_PATH, index=False, float_format="%.8g")
    metadata_df.to_csv(METADATA_PATH, index=False)

    COLUMNS_PATH.write_text(
        json.dumps(
            {
                "feature_set": "amino_acid_composition_plus_dipeptide_composition",
                "n_aac_features": 20,
                "n_dpc_features": 400,
                "n_total_features": 420,
                "amino_acid_order": AMINO_ACIDS,
                "dipeptide_order": DIPEPTIDES,
                "feature_columns": FEATURE_COLUMNS,
                "normalization": {
                    "aac": "count(amino_acid) / sequence_length",
                    "dpc": "count(overlapping_dipeptide) / (sequence_length - 1)",
                },
                "source_dataset": str(INPUT_PATH),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    DOC_PATH.write_text(
        f"""# Ablation A: AAC + DPC Feature Set

## Purpose
This is the non-language-model baseline for Ablation A. It replaces ESM-2
embeddings with deterministic sequence-composition features while retaining
the exact same canonical dataset rows and `sample_id` identifiers.

## Canonical Input
- `{INPUT_PATH}`
- Rows: {len(df)}
- Feature dimensions: 420
  - Amino-acid composition (AAC): 20
  - Dipeptide composition (DPC): 400

## Output Files
- `{FEATURE_PATH}`: numeric matrix only, shape `{feature_df.shape}`.
- `{METADATA_PATH}`: `row_index`, `sample_id`, `species`, and `label`.
- `{COLUMNS_PATH}`: immutable feature names/order and normalization metadata.

## Alignment Rule
The feature-matrix row at `row_index = i` corresponds exactly to the metadata
row with `row_index = i`; `sample_id` is the stable join key. Training and
evaluation code must use the existing frozen split sample IDs, then select
matching rows through `{METADATA_PATH}`. Do not regenerate splits.

## Feature Definitions
For a canonical 20-amino-acid alphabet
`A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y`:

- `aac_X = count(X) / L`
- `dpc_XY = count(overlapping XY) / (L - 1)`

Ambiguous or noncanonical residue characters are excluded before calculating
features. All retained sequences contain at least two canonical residues.
""",
        encoding="utf-8",
    )

    print("PASS — Ablation A features created.")
    print(f"Input rows: {len(df)}")
    print(f"Feature shape: {feature_df.shape}")
    print(f"Feature file: {FEATURE_PATH}")
    print(f"Metadata file: {METADATA_PATH}")
    print(f"Feature schema: {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
