from collections import Counter
from Bio import SeqIO
from scipy.sparse import lil_matrix, save_npz
import pandas as pd
import sys

K = int(sys.argv[1])
TOP_N = 50000

ecoli_fasta = "data\\BVBRC_genome_sequence.fasta"
salmonella_fasta = "data\\selected\\salmonella_fasta.fasta"


def canonical(kmer):
    rc = kmer.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    return min(kmer, rc)


def get_genomes(fasta):
    genomes = {}

    for record in SeqIO.parse(fasta, "fasta"):
        desc = record.description

        if "|" in desc:
            genome_id = desc.split("|")[-1].strip().rstrip("]")

            if genome_id not in genomes:
                genomes[genome_id] = []

            genomes[genome_id].append(str(record.seq).upper())

    return genomes


def count_kmers(genomes):
    counts = Counter()

    for contigs in genomes.values():
        genome_counts = set()

        for seq in contigs:
            for i in range(len(seq) - K + 1):
                kmer = seq[i:i + K]

                if "N" not in kmer:
                    genome_counts.add(canonical(kmer))

        counts.update(genome_counts)

    return counts


def make_matrix(genomes, vocab):
    kmer_to_col = {kmer: i for i, kmer in enumerate(vocab)}

    X = lil_matrix(
        (len(genomes), len(vocab)),
        dtype="float32"
    )

    ids = list(genomes.keys())

    for row, genome_id in enumerate(ids):
        counts = Counter()

        for seq in genomes[genome_id]:
            for i in range(len(seq) - K + 1):
                kmer = seq[i:i + K]

                if "N" not in kmer:
                    kmer = canonical(kmer)

                    if kmer in kmer_to_col:
                        counts[kmer] += 1

        for kmer, count in counts.items():
            X[row, kmer_to_col[kmer]] = count

    return X.tocsr(), ids


print(f"Building k={K}")

ecoli = get_genomes(ecoli_fasta)
salmonella = get_genomes(salmonella_fasta)

print("E. coli genomes:", len(ecoli))
print("Salmonella genomes:", len(salmonella))

print("Counting k-mers...")

ecoli_counts = count_kmers(ecoli)
salmonella_counts = count_kmers(salmonella)

combined = ecoli_counts + salmonella_counts

vocab = [
    kmer
    for kmer, count in combined.most_common(TOP_N)
]

print("Vocabulary:", len(vocab))

X_ecoli, ecoli_ids = make_matrix(ecoli, vocab)
X_salmonella, salmonella_ids = make_matrix(salmonella, vocab)

prefix = f"results/k{K}"

save_npz(
    prefix + "_ecoli.npz",
    X_ecoli
)

save_npz(
    prefix + "_salmonella.npz",
    X_salmonella
)

pd.Series(ecoli_ids).to_csv(
    prefix + "_ecoli_ids.csv",
    index=False,
    header=False
)

pd.Series(salmonella_ids).to_csv(
    prefix + "_salmonella_ids.csv",
    index=False,
    header=False
)

pd.Series(vocab).to_csv(
    prefix + "_vocab.csv",
    index=False,
    header=False
)

print("E. coli:", X_ecoli.shape)
print("Salmonella:", X_salmonella.shape)
print("Saved k-mer sensitivity files")