import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


X_ecoli = load_npz("ecoli_shared.npz")
X_salmonella = load_npz("salmonella_shared.npz")

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

ecoli_ids["Genome ID"] = ecoli_ids["Genome ID"].astype(str)
salmonella_ids["Genome ID"] = salmonella_ids["Genome ID"].astype(str)

ecoli_df = pd.read_csv(
    "data\\selected\\ecolidataset.csv"
)

salmonella_df = pd.read_csv(
    "data\\selected\\salmonelladataset.csv"
)

ecoli_df["Genome ID"] = ecoli_df["Genome ID"].astype(str)
salmonella_df["Genome ID"] = salmonella_df["Genome ID"].astype(str)

ecoli_labels = (
    ecoli_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({
        "Susceptible": 0,
        "Resistant": 1
    })
)

salmonella_labels = (
    salmonella_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({
        "Susceptible": 0,
        "Resistant": 1
    })
)

y_ecoli = (
    ecoli_ids["Genome ID"]
    .map(ecoli_labels)
    .values
)

y_salmonella = (
    salmonella_ids["Genome ID"]
    .map(salmonella_labels)
    .values
)

assert not pd.isna(y_ecoli).any()
assert not pd.isna(y_salmonella).any()

y_ecoli = y_ecoli.astype(int)
y_salmonella = y_salmonella.astype(int)

features = pd.read_csv(
    "shared_kmers.csv"
)["kmer"].astype(str).tolist()

assert len(features) == X_ecoli.shape[1]
assert len(features) == X_salmonella.shape[1]

os.makedirs("results", exist_ok=True)

results = []
top_kmer_results = []

for seed in range(30):

    rng = np.random.default_rng(seed)

    shuffled_ecoli = y_ecoli.copy()
    shuffled_salmonella = y_salmonella.copy()

    rng.shuffle(shuffled_ecoli)
    rng.shuffle(shuffled_salmonella)

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

    xgb.fit(
        X_ecoli,
        shuffled_ecoli
    )

    pred = xgb.predict_proba(
        X_salmonella
    )[:, 1]

    auc = roc_auc_score(
        y_salmonella,
        pred
    )

    results.append({
        "direction": "E_coli_to_Salmonella",
        "model": "XGBoost",
        "seed": seed,
        "AUROC": auc,
        "label_type": "shuffled"
    })

    importance = xgb.feature_importances_

    top_indices = np.argsort(
        importance
    )[::-1][:500]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "XGBoost",
            "direction": "E_coli_to_Salmonella",
            "seed": seed,
            "rank": rank,
            "label_type": "shuffled"
        })

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(
        X_ecoli,
        shuffled_ecoli
    )

    pred = rf.predict_proba(
        X_salmonella
    )[:, 1]

    auc = roc_auc_score(
        y_salmonella,
        pred
    )

    results.append({
        "direction": "E_coli_to_Salmonella",
        "model": "RandomForest",
        "seed": seed,
        "AUROC": auc,
        "label_type": "shuffled"
    })

    importance = rf.feature_importances_

    top_indices = np.argsort(
        importance
    )[::-1][:500]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "RandomForest",
            "direction": "E_coli_to_Salmonella",
            "seed": seed,
            "rank": rank,
            "label_type": "shuffled"
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

    xgb.fit(
        X_salmonella,
        shuffled_salmonella
    )

    pred = xgb.predict_proba(
        X_ecoli
    )[:, 1]

    auc = roc_auc_score(
        y_ecoli,
        pred
    )

    results.append({
        "direction": "Salmonella_to_E_coli",
        "model": "XGBoost",
        "seed": seed,
        "AUROC": auc,
        "label_type": "shuffled"
    })

    importance = xgb.feature_importances_

    top_indices = np.argsort(
        importance
    )[::-1][:500]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "XGBoost",
            "direction": "Salmonella_to_E_coli",
            "seed": seed,
            "rank": rank,
            "label_type": "shuffled"
        })

    rf = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        n_jobs=-1,
        random_state=seed
    )

    rf.fit(
        X_salmonella,
        shuffled_salmonella
    )

    pred = rf.predict_proba(
        X_ecoli
    )[:, 1]

    auc = roc_auc_score(
        y_ecoli,
        pred
    )

    results.append({
        "direction": "Salmonella_to_E_coli",
        "model": "RandomForest",
        "seed": seed,
        "AUROC": auc,
        "label_type": "shuffled"
    })

    importance = rf.feature_importances_

    top_indices = np.argsort(
        importance
    )[::-1][:500]

    for rank, index in enumerate(
        top_indices,
        start=1
    ):
        top_kmer_results.append({
            "kmer": features[index],
            "importance": importance[index],
            "model": "RandomForest",
            "direction": "Salmonella_to_E_coli",
            "seed": seed,
            "rank": rank,
            "label_type": "shuffled"
        })

    print(
        f"Finished shuffle {seed + 1}/30"
    )


results = pd.DataFrame(results)

results.to_csv(
    "results/shuffle_control.csv",
    index=False
)

top_kmer_results = pd.DataFrame(
    top_kmer_results
)

top_kmer_results.to_csv(
    "results/shuffled_top_kmers.csv",
    index=False
)

summary = (
    results
    .groupby(
        ["direction", "model"]
    )["AUROC"]
    .agg([
        "mean",
        "std",
        "min",
        "max"
    ])
)

print("\nSHUFFLED AUROC")
print(summary)

print("\nSaved:")
print("results/shuffle_control.csv")
print("results/shuffled_top_kmers.csv")
