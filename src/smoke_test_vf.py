
import pandas as pd
from embed import embed_species
from train_vf import train_and_eval


df_pos = pd.read_csv("data/vf_proteins.csv")
df_neg = pd.read_csv("data/vf_negatives.csv")
df_combined = pd.concat([df_pos, df_neg], ignore_index=True)


species = "Escherichia coli"
X = embed_species(df_combined, species)
y = df_combined[df_combined["species"] == species]["label"].values

auroc, model = train_and_eval(X, y, seed=0)
print(f"Smoke test AUROC ({species}): {auroc:.3f}")
