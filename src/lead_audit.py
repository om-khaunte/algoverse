import pandas as pd
import json
from difflib import SequenceMatcher


def sequence_identity(seq1, seq2):
    """Similarity ratio between two sequences, from 0 (nothing alike) to 1 (identical)."""
    return SequenceMatcher(None, seq1, seq2).ratio()


def leak_audit_sequences(source_seqs, target_seqs, sample_size=50):
    """
    Checks max pairwise sequence identity between two sets of sequences.
    NOTE: uses a simplified string-similarity check and a sample for speed.
    For a rigorous audit, use a proper alignment tool (e.g. BLAST) on full sets.
    """
    source_sample = source_seqs[:sample_size]
    target_sample = target_seqs[:sample_size]
    max_identity = 0
    for s in source_sample:
        for t in target_sample:
            identity = sequence_identity(s, t)
            if identity > max_identity:
                max_identity = identity
    return max_identity


def audit_fold(df_full, fold, sample_size=50):
    """Runs the leak audit on a single LOSO fold using frozen sample IDs."""
    test_ids = fold["test_sample_ids"]
    train_ids = fold["train_sample_ids"]
    test_seqs = df_full[df_full["sample_id"].isin(test_ids)]["sequence"].tolist()
    train_seqs = df_full[df_full["sample_id"].isin(train_ids)]["sequence"].tolist()
    max_identity = leak_audit_sequences(train_seqs, test_seqs, sample_size=sample_size)
    print(f"Fold {fold['fold_id']} (test: {fold['test_species']}): "
          f"train={len(train_seqs)}, test={len(test_seqs)}, max pairwise identity = {max_identity:.3f}")
    return max_identity


if __name__ == "__main__":
    df_full = pd.read_csv("data/vf_dataset_cdhit40_zero_shot_ready.csv")
    with open("splits/loso_folds.json") as f:
        loso_folds = json.load(f)
    audit_fold(df_full, loso_folds["folds"][0])
