import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.sparse import csr_matrix, save_npz


def canonical_kmer(kmer):
    complement = str.maketrans("ACGT", "TGCA")
    reverse_complement = kmer.translate(complement)[::-1]
    return min(kmer, reverse_complement)


def genome_id_from_record(record):
    # Example:
    # [Escherichia coli strain AMC_351 | 562.64824]
    description = record.description

    if "|" not in description:
        return None

    genome_id = description.split("|")[-1].strip().rstrip("]")

    return genome_id


def count_genome_kmers(fasta_file, k):
    genome_kmers = defaultdict(Counter)

    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_id = genome_id_from_record(record)

        if genome_id is None:
            continue

        sequence = str(record.seq).upper()

        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i + k]

            if "N" in kmer:
                continue

            genome_kmers[genome_id][canonical_kmer(kmer)] += 1

    return genome_kmers


def build_matrix(fasta_file, output, k=10, top_n=50000):
    print("Reading FASTA...")
    genome_kmers = count_genome_kmers(fasta_file, k)

    print(f"Found {len(genome_kmers)} genomes")

    # Count how many genomes contain each k-mer.
    # This prevents a k-mer repeated many times in one genome
    # from dominating the feature selection.
    genome_frequency = Counter()

    for counts in genome_kmers.values():
        genome_frequency.update(counts.keys())

    features = [
        kmer for kmer, count in genome_frequency.most_common(top_n)
    ]

    feature_index = {
        kmer: i for i, kmer in enumerate(features)
    }

    genome_ids = sorted(genome_kmers)

    rows = []
    cols = []
    data = []

    for row, genome_id in enumerate(genome_ids):
        counts = genome_kmers[genome_id]

        for kmer, count in counts.items():
            if kmer in feature_index:
                rows.append(row)
                cols.append(feature_index[kmer])
                data.append(count)

    X = csr_matrix(
        (data, (rows, cols)),
        shape=(len(genome_ids), len(features)),
        dtype=np.float32
    )

    save_npz(output + ".npz", X)

    pd.Series(genome_ids).to_csv(
        output + "_genome_ids.csv",
        index=False,
        header=False
    )

    pd.Series(features).to_csv(
        output + "_features.csv",
        index=False,
        header=False
    )

    print("Done!")
    print("Matrix shape:", X.shape)
    print("Non-zero entries:", X.nnz)
    print("Saved:", output + ".npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=50000)

    args = parser.parse_args()

    build_matrix(
        args.fasta,
        args.output,
        args.k,
        args.top_n
    )