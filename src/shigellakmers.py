import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

X_ecoli = load_npz("ecoli_shared.npz")
X_shigella = load_npz("shigella_shared.npz")

ecoli_ids = pd.read_csv(
    "ecoli_shared_ids.csv",
    header=None,
    names=["Genome ID"]
)

shigella_ids = pd.read_csv(
    "shigella_shared_ids.csv",
    header=None,
    names=["Genome ID"]
)

ecoli_ids["Genome ID"] = ecoli_ids["Genome ID"].astype(str)
shigella_ids["Genome ID"] = shigella_ids["Genome ID"].astype(str)

ecoli_df = pd.read_csv(
    "data\\selected\\ecolidataset.csv"
)

shigella_df = pd.read_csv(
    "data\\selected\\BVBRC_genome_amr.csv"
)

ecoli_df["Genome ID"] = ecoli_df["Genome ID"].astype(str)
shigella_df["Genome ID"] = shigella_df["Genome ID"].astype(str)

ecoli_labels = (
    ecoli_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({
        "Susceptible": 0,
        "Resistant": 1
    })
)

shigella_labels = (
    shigella_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({
        "Susceptible": 0,
        "Resistant": 1
    })
)

y_ecoli = ecoli_ids["Genome ID"].map(ecoli_labels).values
y_shigella = shigella_ids["Genome ID"].map(shigella_labels).values

assert not pd.isna(y_ecoli).any()
assert not pd.isna(y_shigella).any()

y_ecoli = y_ecoli.astype(int)
y_shigella = y_shigella.astype(int)

print("E. coli:", X_ecoli.shape)
print("Shigella:", X_shigella.shape)
print("E. coli labels:", np.bincount(y_ecoli))
print("Shigella labels:", np.bincount(y_shigella))

results = []

for seed in [42, 43, 44]:

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed
    )

    xgb.fit(X_ecoli, y_ecoli)

    pred = xgb.predict_proba(X_shigella)[:, 1]

    results.append({
        "direction": "E_coli_to_Shigella",
        "model": "XGBoost",
        "seed": seed,
        "AUROC": roc_auc_score(y_shigella, pred)
    })

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_ecoli, y_ecoli)

    pred = rf.predict_proba(X_shigella)[:, 1]

    results.append({
        "direction": "E_coli_to_Shigella",
        "model": "RandomForest",
        "seed": seed,
        "AUROC": roc_auc_score(y_shigella, pred)
    })

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed
    )

    xgb.fit(X_shigella, y_shigella)

    pred = xgb.predict_proba(X_ecoli)[:, 1]

    results.append({
        "direction": "Shigella_to_E_coli",
        "model": "XGBoost",
        "seed": seed,
        "AUROC": roc_auc_score(y_ecoli, pred)
    })

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(X_shigella, y_shigella)

    pred = rf.predict_proba(X_ecoli)[:, 1]

    results.append({
        "direction": "Shigella_to_E_coli",
        "model": "RandomForest",
        "seed": seed,
        "AUROC": roc_auc_score(y_ecoli, pred)
    })

    print(f"Finished seed {seed}")

results = pd.DataFrame(results)

results.to_csv(
    "results/ecoli_shigella_transfer.csv",
    index=False
)

summary = (
    results
    .groupby(["direction", "model"])["AUROC"]
    .agg(["mean", "std", "min", "max"])
)

print("\n===== E. COLI / SHIGELLA TRANSFER =====")
print(summary)

print("\nSaved:")
print("results/ecoli_shigella_transfer.csv")