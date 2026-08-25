import os
import re
import tarfile
import pandas as pd
from collections import defaultdict

TOP_KMERS_FILE = "results\\shuffled_top_kmers.csv"
CARD_ARCHIVE = "C:\\Users\\Arnav\\Downloads\\card-data.tar.bz2"
OUTPUT_DIR = "results/card_mapping_shuffled"
TOP_N = 500
K = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(TOP_KMERS_FILE):
    raise FileNotFoundError(f"Could not find {TOP_KMERS_FILE}")

if not os.path.exists(CARD_ARCHIVE):
    raise FileNotFoundError(f"Could not find {CARD_ARCHIVE}")

CARD_DIR = os.path.join(OUTPUT_DIR, "card_data")

if not os.path.exists(CARD_DIR):
    os.makedirs(CARD_DIR)
    with tarfile.open(CARD_ARCHIVE, "r:bz2") as tar:
        tar.extractall(CARD_DIR)

aro_index_path = None

for root, dirs, files in os.walk(CARD_DIR):
    if "aro_index.tsv" in files:
        aro_index_path = os.path.join(root, "aro_index.tsv")
        break

if aro_index_path is None:
    raise FileNotFoundError("Could not find aro_index.tsv")

CARD_ROOT = os.path.dirname(aro_index_path)

fasta_files = []

for filename in os.listdir(CARD_ROOT):
    if filename.startswith("nucleotide_fasta_") and filename.endswith(".fasta"):
        fasta_files.append(os.path.join(CARD_ROOT, filename))

if len(fasta_files) == 0:
    raise FileNotFoundError("No CARD nucleotide FASTA files found")

aro_index = pd.read_csv(
    aro_index_path,
    sep="\t",
    dtype=str
)

aro_lookup = {}

for _, row in aro_index.iterrows():
    aro = row.get("ARO Accession")

    if pd.isna(aro):
        continue

    aro_lookup[str(aro)] = {
        "model_name": row.get("Model Name", ""),
        "aro_name": row.get("ARO Name", ""),
        "amr_gene_family": row.get("AMR Gene Family", ""),
        "drug_class": row.get("Drug Class", ""),
        "mechanism": row.get("Resistance Mechanism", ""),
        "card_short_name": row.get("CARD Short Name", "")
    }


def read_fasta(filename):
    records = []
    header = None
    sequence_parts = []

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):
                if header is not None:
                    records.append((
                        header,
                        "".join(sequence_parts).upper()
                    ))

                header = line[1:]
                sequence_parts = []

            else:
                sequence_parts.append(line)

    if header is not None:
        records.append((
            header,
            "".join(sequence_parts).upper()
        ))

    return records


def get_aro_from_header(header):
    match = re.search(r"(ARO:\d+)", header)

    if match:
        return match.group(1)

    return None


card_records = []

for fasta_file in fasta_files:
    records = read_fasta(fasta_file)

    for header, sequence in records:
        card_records.append({
            "header": header,
            "sequence": sequence,
            "aro": get_aro_from_header(header),
            "source_file": os.path.basename(fasta_file)
        })

print("CARD nucleotide sequences:", len(card_records))

top_kmers = pd.read_csv(TOP_KMERS_FILE)

required_columns = {
    "kmer",
    "importance",
    "model",
    "direction",
    "seed"
}

missing = required_columns - set(top_kmers.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

top_kmers["kmer"] = (
    top_kmers["kmer"]
    .astype(str)
    .str.upper()
    .str.strip()
)

valid_bases = set("ACGT")

top_kmers = top_kmers[
    top_kmers["kmer"].apply(
        lambda x: len(x) == K and set(x).issubset(valid_bases)
    )
].copy()

top_groups = (
    top_kmers
    .sort_values(
        ["direction", "model", "seed", "importance"],
        ascending=[True, True, True, False]
    )
    .groupby(
        ["direction", "model", "seed"],
        sort=False
    )
    .head(TOP_N)
    .copy()
)

model_kmers = set(top_groups["kmer"].unique())

card_kmer_hits = defaultdict(list)

for record_number, record in enumerate(card_records):
    sequence = record["sequence"]

    for i in range(len(sequence) - K + 1):
        kmer = sequence[i:i + K]

        if kmer in model_kmers:
            card_kmer_hits[kmer].append(record_number)

print("Unique top k-mers:", len(model_kmers))
print("Top k-mers with CARD matches:", len(card_kmer_hits))

match_rows = []

for _, row in top_groups.iterrows():
    kmer = row["kmer"]
    hits = card_kmer_hits.get(kmer, [])

    if len(hits) == 0:
        match_rows.append({
            "kmer": kmer,
            "importance": row["importance"],
            "model": row["model"],
            "direction": row["direction"],
            "seed": row["seed"],
            "CARD_match": False,
            "ARO_accession": "",
            "CARD_short_name": "",
            "ARO_name": "",
            "AMR_gene_family": "",
            "Drug_class": "",
            "Resistance_mechanism": "",
            "CARD_sequence_header": ""
        })
        continue

    seen = set()

    for hit_number in hits:
        record = card_records[hit_number]
        aro = record["aro"]

        annotation = aro_lookup.get(
            aro,
            {
                "aro_name": "",
                "amr_gene_family": "",
                "drug_class": "",
                "mechanism": "",
                "card_short_name": ""
            }
        )

        key = (kmer, aro, record["header"])

        if key in seen:
            continue

        seen.add(key)

        match_rows.append({
            "kmer": kmer,
            "importance": row["importance"],
            "model": row["model"],
            "direction": row["direction"],
            "seed": row["seed"],
            "CARD_match": True,
            "ARO_accession": aro or "",
            "CARD_short_name": annotation["card_short_name"],
            "ARO_name": annotation["aro_name"],
            "AMR_gene_family": annotation["amr_gene_family"],
            "Drug_class": annotation["drug_class"],
            "Resistance_mechanism": annotation["mechanism"],
            "CARD_sequence_header": record["header"]
        })

matches = pd.DataFrame(match_rows)

matches_file = os.path.join(
    OUTPUT_DIR,
    "card_kmer_matches.csv"
)

matches.to_csv(matches_file, index=False)

summary_rows = []

for (direction, model, seed), group in top_groups.groupby(
    ["direction", "model", "seed"]
):
    kmers = set(group["kmer"])

    matching_kmers = sum(
        kmer in card_kmer_hits
        for kmer in kmers
    )

    summary_rows.append({
        "direction": direction,
        "model": model,
        "seed": seed,
        "top_kmers": len(kmers),
        "CARD_matching_kmers": matching_kmers,
        "CARD_fraction": matching_kmers / len(kmers)
    })

summary = pd.DataFrame(summary_rows)

summary_file = os.path.join(
    OUTPUT_DIR,
    "card_summary.csv"
)

summary.to_csv(summary_file, index=False)

seed_summary = (
    summary
    .groupby(["direction", "model"])["CARD_fraction"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

seed_summary_file = os.path.join(
    OUTPUT_DIR,
    "card_seed_summary.csv"
)

seed_summary.to_csv(seed_summary_file, index=False)

matched_only = matches[
    matches["CARD_match"] == True
].copy()

gene_counts = (
    matched_only
    .groupby([
        "direction",
        "model",
        "CARD_short_name",
        "ARO_accession",
        "ARO_name",
        "AMR_gene_family",
        "Drug_class",
        "Resistance_mechanism"
    ])
    .agg(
        matching_kmers=("kmer", "nunique")
    )
    .reset_index()
    .sort_values(
        "matching_kmers",
        ascending=False
    )
)

gene_counts_file = os.path.join(
    OUTPUT_DIR,
    "card_gene_hits.csv"
)

gene_counts.to_csv(gene_counts_file, index=False)

print("\nCARD MAPPING SUMMARY")
print(summary.to_string(index=False))

print("\nCARD FRACTION ACROSS SEEDS")
print(seed_summary.to_string(index=False))

print("\nTOP CARD GENE HITS")
print(gene_counts.head(20).to_string(index=False))

print("\nSaved:")
print(matches_file)
print(summary_file)
print(seed_summary_file)
print(gene_counts_file)