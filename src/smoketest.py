import pandas as pd
import numpy as np

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# Load k-mer matrix
X = load_npz("ecoli_k10.npz")

# Load genome IDs in the same order as X
kmer_ids = pd.read_csv(
    "ecoli_k10_genome_ids.csv",
    header=None,
    names=["Genome ID"]
)

# Load phenotype data
df = pd.read_csv("data\\selected\\ecolidataset.csv")

# Make IDs consistent
df["Genome ID"] = df["Genome ID"].astype(str)
kmer_ids["Genome ID"] = kmer_ids["Genome ID"].astype(str)

# Match labels to k-mer rows
labels = kmer_ids.merge(
    df[["Genome ID", "Resistant Phenotype"]],
    on="Genome ID",
    how="left"
)

# Convert Resistant/Susceptible to 1/0
y = (
    labels["Resistant Phenotype"]
    .map({
        "Susceptible": 0,
        "Resistant": 1
    })
    .values
)

# Check everything lines up
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Missing labels:", np.isnan(y).sum())

# 80/20 held-out E. coli split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training genomes:", X_train.shape[0])
print("Test genomes:", X_test.shape[0])

# XGBoost
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42
)

model.fit(X_train, y_train)

# Predict probability of Resistant
y_prob = model.predict_proba(X_test)[:, 1]

# AUROC
auroc = roc_auc_score(y_test, y_prob)

print(f"\nE. coli held-out AUROC: {auroc:.4f}")