import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


# -----------------------------
# Files
# -----------------------------

ECOLI_X = "ecoli_shared.npz"
SALMONELLA_X = "salmonella_shared.npz"

ECOLI_IDS = "ecoli_shared_ids.csv"
SALMONELLA_IDS = "salmonella_shared_ids.csv"

ECOLI_CSV = "data\\selected\\ecolidataset.csv"
SALMONELLA_CSV = "data\\selected\\salmonelladataset.csv"

KMER_FILE = "shared_kmers.csv"


# -----------------------------
# Load data
# -----------------------------

X_ecoli = load_npz(ECOLI_X)
X_salmonella = load_npz(SALMONELLA_X)

ecoli_ids = pd.read_csv(
    ECOLI_IDS, header=None, names=["Genome ID"]
)

salmonella_ids = pd.read_csv(
    SALMONELLA_IDS, header=None, names=["Genome ID"]
)

ecoli_ids["Genome ID"] = ecoli_ids["Genome ID"].astype(str)
salmonella_ids["Genome ID"] = salmonella_ids["Genome ID"].astype(str)

ecoli_df = pd.read_csv(ECOLI_CSV)
salmonella_df = pd.read_csv(SALMONELLA_CSV)

ecoli_df["Genome ID"] = ecoli_df["Genome ID"].astype(str)
salmonella_df["Genome ID"] = salmonella_df["Genome ID"].astype(str)


# -----------------------------
# Attach labels
# -----------------------------

def get_labels(ids, df):
    label_map = (
        df.drop_duplicates("Genome ID")
        .set_index("Genome ID")["Resistant Phenotype"]
        .map({
            "Susceptible": 0,
            "Resistant": 1
        })
    )

    y = ids["Genome ID"].map(label_map)

    return y.values


y_ecoli_all = get_labels(ecoli_ids, ecoli_df)
y_salmonella_all = get_labels(salmonella_ids, salmonella_df)


# Keep only genomes with labels
ecoli_keep = ~pd.isna(y_ecoli_all)
salmonella_keep = ~pd.isna(y_salmonella_all)

X_ecoli = X_ecoli[ecoli_keep]
X_salmonella = X_salmonella[salmonella_keep]

y_ecoli = y_ecoli_all[ecoli_keep].astype(int)
y_salmonella = y_salmonella_all[salmonella_keep].astype(int)

ecoli_ids = ecoli_ids[ecoli_keep].reset_index(drop=True)
salmonella_ids = salmonella_ids[salmonella_keep].reset_index(drop=True)


print("E. coli labeled genomes:", len(y_ecoli))
print("  Resistant:", sum(y_ecoli == 1))
print("  Susceptible:", sum(y_ecoli == 0))

print("Salmonella labeled genomes:", len(y_salmonella))
print("  Resistant:", sum(y_salmonella == 1))
print("  Susceptible:", sum(y_salmonella == 0))


# -----------------------------
# Models
# -----------------------------

def make_xgb(seed):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed
    )


def make_rf(seed):
    return RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )


# -----------------------------
# Zero-shot experiments
# -----------------------------

seeds = [42, 123, 2024]

results = []


def run_direction(
    train_X,
    train_y,
    test_X,
    test_y,
    direction,
    shuffled=False
):

    for seed in seeds:

        rng = np.random.default_rng(seed)

        y_train = train_y.copy()

        if shuffled:
            rng.shuffle(y_train)

        models = {
            "XGBoost": make_xgb(seed),
            "RandomForest": make_rf(seed)
        }

        for model_name, model in models.items():

            model.fit(train_X, y_train)

            probabilities = model.predict_proba(test_X)[:, 1]

            auc = roc_auc_score(
                test_y,
                probabilities
            )

            results.append({
                "direction": direction,
                "model": model_name,
                "seed": seed,
                "AUROC": auc,
                "label_shuffled": shuffled
            })

            print(
                f"{direction} | "
                f"{model_name} | "
                f"seed={seed} | "
                f"AUROC={auc:.4f}"
            )


# E. coli -> Salmonella
run_direction(
    X_ecoli,
    y_ecoli,
    X_salmonella,
    y_salmonella,
    "E_coli_to_Salmonella"
)


# Salmonella -> E. coli
run_direction(
    X_salmonella,
    y_salmonella,
    X_ecoli,
    y_ecoli,
    "Salmonella_to_E_coli"
)


# -----------------------------
# Label-shuffled controls
# -----------------------------

run_direction(
    X_ecoli,
    y_ecoli,
    X_salmonella,
    y_salmonella,
    "E_coli_to_Salmonella",
    shuffled=True
)

run_direction(
    X_salmonella,
    y_salmonella,
    X_ecoli,
    y_ecoli,
    "Salmonella_to_E_coli",
    shuffled=True
)


# -----------------------------
# Save results
# -----------------------------

os.makedirs("results", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "results/zero_shot_results.csv",
    index=False
)

print("\n===== SUMMARY =====")

summary = (
    results_df
    .groupby(
        ["direction", "model", "label_shuffled"]
    )["AUROC"]
    .agg(["mean", "std"])
)

print(summary)

print("\nSaved:")
print("results/zero_shot_results.csv")