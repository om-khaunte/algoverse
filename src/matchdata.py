import pandas as pd
from scipy.sparse import load_npz

# CHANGE THESE TWO FILENAMES
phenotypes = pd.read_csv("data\\selected\\ecolidataset.csv")
X = load_npz("ecoli_k10.npz")

# IDs in the same order as the rows of X
kmer_ids = pd.read_csv(
    "ecoli_k10_genome_ids.csv",
    header=None,
    names=["Genome ID"]
)

# Make IDs consistent
phenotypes["Genome ID"] = phenotypes["Genome ID"].astype(str)
kmer_ids["Genome ID"] = kmer_ids["Genome ID"].astype(str)

# Match k-mer genomes to phenotype records
merged = kmer_ids.merge(
    phenotypes[["Genome ID", "Antibiotic", "Resistant Phenotype"]],
    on="Genome ID",
    how="inner"
)

print("X shape:", X.shape)
print("K-mer genomes:", len(kmer_ids))
print("Matched genomes:", len(merged))

print("\nAntibiotics:")
print(merged["Antibiotic"].value_counts())

print("\nPhenotypes:")
print(merged["Resistant Phenotype"].value_counts())