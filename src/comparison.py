from collections import Counter
from Bio import SeqIO
from scipy.sparse import lil_matrix, save_npz
import pandas as pd

K = 10

fasta = "data\\selected\\genomes.fasta"
csv_file = "data\\selected\\BVBRC_genome_amr.csv"

df = pd.read_csv(csv_file)
df["Genome ID"] = df["Genome ID"].astype(str).str.strip()

labeled_ids = set(
    df["Genome ID"]
    .dropna()
)

vocab = pd.read_csv(
    "shared_kmers.csv"
)["kmer"].astype(str).tolist()

kmer_to_col = {
    kmer: i
    for i, kmer in enumerate(vocab)
}

genomes = {}

for record in SeqIO.parse(fasta, "fasta"):
    genome_id = record.id.strip()

    if genome_id not in labeled_ids:
        continue

    if genome_id not in genomes:
        genomes[genome_id] = []

    genomes[genome_id].append(
        str(record.seq).upper()
    )

print("Labeled Shigella genomes found:", len(genomes))

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

            if "N" in kmer:
                continue

            rc = kmer.translate(
                str.maketrans("ACGT", "TGCA")
            )[::-1]

            kmer = min(kmer, rc)

            if kmer in kmer_to_col:
                counts[kmer] += 1

    for kmer, count in counts.items():
        X[row, kmer_to_col[kmer]] = count

X = X.tocsr()

save_npz(
    "shigella_shared.npz",
    X
)

pd.DataFrame({
    "Genome ID": ids
}).to_csv(
    "shigella_shared_ids.csv",
    index=False,
    header=False
)

print("Shigella matrix:", X.shape)
print("Non-zero entries:", X.nnz)
print("Saved: shigella_shared.npz")
print("Saved: shigella_shared_ids.csv")