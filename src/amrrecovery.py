import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

X_ecoli = load_npz("ecoli_shared.npz")
X_salmonella = load_npz("salmonella_shared.npz")
X_enterobacter = load_npz("enterobacter_shared.npz")

ecoli_ids = pd.read_csv(
    "ecoli_shared_ids.csv",
    header=None,
    names=["Genome ID"]
)

salmonella_ids = pd.read_csv(
    "salmonella_shared_ids.csv",
    header=None,
    names=["Genome ID"]
)

enterobacter_ids = pd.read_csv(
    "enterobacter_shared_ids.csv",
    header=None,
    names=["Genome ID"]
)

for ids in [ecoli_ids, salmonella_ids, enterobacter_ids]:
    ids["Genome ID"] = ids["Genome ID"].astype(str)

ecoli_df = pd.read_csv("data\\selected\\ecolidataset.csv")
salmonella_df = pd.read_csv("data\\selected\\salmonelladataset.csv")
enterobacter_df = pd.read_csv("data\\selected\\enterobacterdataset.csv")

for df in [ecoli_df, salmonella_df, enterobacter_df]:
    df["Genome ID"] = df["Genome ID"].astype(str)

def make_label_lookup(df):
    return (
        df
        .drop_duplicates("Genome ID")
        .set_index("Genome ID")["Resistant Phenotype"]
        .map({
            "Susceptible": 0,
            "Resistant": 1
        })
    )

ecoli_labels = make_label_lookup(ecoli_df)
salmonella_labels = make_label_lookup(salmonella_df)
enterobacter_labels = make_label_lookup(enterobacter_df)

y_ecoli = ecoli_ids["Genome ID"].map(ecoli_labels).values
y_salmonella = salmonella_ids["Genome ID"].map(salmonella_labels).values
y_enterobacter = enterobacter_ids["Genome ID"].map(enterobacter_labels).values

assert not pd.isna(y_ecoli).any()
assert not pd.isna(y_salmonella).any()
assert not pd.isna(y_enterobacter).any()

y_ecoli = y_ecoli.astype(int)
y_salmonella = y_salmonella.astype(int)
y_enterobacter = y_enterobacter.astype(int)

features = pd.read_csv(
    "shared_kmers.csv"
)["kmer"].astype(str).tolist()

assert len(features) == X_ecoli.shape[1]
assert len(features) == X_salmonella.shape[1]
assert len(features) == X_enterobacter.shape[1]

os.makedirs("results", exist_ok=True)

results = []
top_kmer_results = []

def run_xgb(X_train, y_train, X_test, y_test, source, target, seed):
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, pred)

    results.append({
        "direction": f"{source}_to_{target}",
        "model": "XGBoost",
        "seed": seed,
        "AUROC": auc,
        "label_type": "real"
    })

    importance = model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:500]

    for rank, index in enumerate(top_indices, start=1):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "XGBoost",
            "direction": f"{source}_to_{target}",
            "seed": seed,
            "rank": rank,
            "label_type": "real"
        })

def run_rf(X_train, y_train, X_test, y_test, source, target, seed):
    model = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )

    model.fit(X_train, y_train)

    pred = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, pred)

    results.append({
        "direction": f"{source}_to_{target}",
        "model": "RandomForest",
        "seed": seed,
        "AUROC": auc,
        "label_type": "real"
    })

    importance = model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:500]

    for rank, index in enumerate(top_indices, start=1):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "RandomForest",
            "direction": f"{source}_to_{target}",
            "seed": seed,
            "rank": rank,
            "label_type": "real"
        })

for seed in range(30):

    run_xgb(
        X_ecoli,
        y_ecoli,
        X_salmonella,
        y_salmonella,
        "E_coli",
        "Salmonella",
        seed
    )

    run_rf(
        X_ecoli,
        y_ecoli,
        X_salmonella,
        y_salmonella,
        "E_coli",
        "Salmonella",
        seed
    )

    run_xgb(
        X_salmonella,
        y_salmonella,
        X_ecoli,
        y_ecoli,
        "Salmonella",
        "E_coli",
        seed
    )

    run_rf(
        X_salmonella,
        y_salmonella,
        X_ecoli,
        y_ecoli,
        "Salmonella",
        "E_coli",
        seed
    )

    run_xgb(
        X_ecoli,
        y_ecoli,
        X_enterobacter,
        y_enterobacter,
        "E_coli",
        "Enterobacter",
        seed
    )

    run_rf(
        X_ecoli,
        y_ecoli,
        X_enterobacter,
        y_enterobacter,
        "E_coli",
        "Enterobacter",
        seed
    )

    run_xgb(
        X_enterobacter,
        y_enterobacter,
        X_ecoli,
        y_ecoli,
        "Enterobacter",
        "E_coli",
        seed
    )

    run_rf(
        X_enterobacter,
        y_enterobacter,
        X_ecoli,
        y_ecoli,
        "Enterobacter",
        "E_coli",
        seed
    )

    print(f"Finished seed {seed + 1}/30")

results = pd.DataFrame(results)

results.to_csv(
    "results/species_transfer_real.csv",
    index=False
)

top_kmer_results = pd.DataFrame(top_kmer_results)

top_kmer_results.to_csv(
    "results/species_transfer_top_kmers.csv",
    index=False
)

summary = (
    results
    .groupby(["direction", "model"])["AUROC"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
)

summary.to_csv(
    "results/species_transfer_summary.csv",
    index=False
)

print("\nSPECIES-HELD-OUT TRANSFER")
print(summary.to_string(index=False))

print("\nSaved:")
print("results/species_transfer_real.csv")
print("results/species_transfer_top_kmers.csv")
print("results/species_transfer_summary.csv")