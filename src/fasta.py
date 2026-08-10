import pandas as pd
from Bio import SeqIO

csv = pd.read_csv("data\\selected\\ecolidataset.csv")

fasta_ids = set()

for record in SeqIO.parse("C:\\Users\\Arnav\\Downloads\\AMR_project\\data\\BVBRC_genome_sequence.fasta", "fasta"):
    if "|" in record.description:
        fasta_ids.add(
            record.description.split("|")[-1].strip().rstrip("]")
        )

csv_ids = set(csv["Genome ID"].astype(str))

matched = csv_ids & fasta_ids

print("CSV genomes:", len(csv_ids))
print("FASTA genomes:", len(fasta_ids))
print("Matching genomes:", len(matched))