import pandas as pd
from Bio import SeqIO

csv = pd.read_csv("data\\selected\\salmonelladataset.csv")

csv_ids = set(csv["Genome ID"].astype(str))

fasta_ids = set()

for record in SeqIO.parse("data\\selected\\lol.fasta", "fasta"):
    description = record.description

    if "|" in description:
        genome_id = description.split("|")[-1].strip().rstrip("]")
        fasta_ids.add(genome_id)

matched = csv_ids & fasta_ids

print("CSV unique IDs:", len(csv_ids))
print("FASTA unique IDs:", len(fasta_ids))
print("Matching IDs:", len(matched))

print("\nSome matching IDs:")
print(sorted(matched)[:20])

print("\nSome CSV IDs NOT in FASTA:")
print(sorted(csv_ids - fasta_ids)[:20])

matched_csv = csv[csv["Genome ID"].astype(str).isin(matched)]

print("Matched:", len(matched_csv))
print(matched_csv["Resistant Phenotype"].value_counts())