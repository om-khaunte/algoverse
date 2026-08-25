import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


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

results = []

for k in [8, 12]:

    X_ecoli = load_npz(
        f"results/k{k}_ecoli.npz"
    )

    X_salmonella = load_npz(
        f"results/k{k}_salmonella.npz"
    )

    ecoli_ids = pd.read_csv(
        f"results/k{k}_ecoli_ids.csv",
        header=None
    )[0].astype(str)

    salmonella_ids = pd.read_csv(
        f"results/k{k}_salmonella_ids.csv",
        header=None
    )[0].astype(str)

    y_ecoli = ecoli_ids.map(ecoli_labels).values
    y_salmonella = salmonella_ids.map(salmonella_labels).values

    assert not pd.isna(y_ecoli).any()
    assert not pd.isna(y_salmonella).any()

    y_ecoli = y_ecoli.astype(int)
    y_salmonella = y_salmonella.astype(int)

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

        pred = xgb.predict_proba(
            X_salmonella
        )[:, 1]

        results.append({
            "k": k,
            "direction": "E_coli_to_Salmonella",
            "model": "XGBoost",
            "seed": seed,
            "AUROC": roc_auc_score(
                y_salmonella,
                pred
            )
        })

        rf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            n_jobs=-1,
            random_state=seed
        )

        rf.fit(X_ecoli, y_ecoli)

        pred = rf.predict_proba(
            X_salmonella
        )[:, 1]

        results.append({
            "k": k,
            "direction": "E_coli_to_Salmonella",
            "model": "RandomForest",
            "seed": seed,
            "AUROC": roc_auc_score(
                y_salmonella,
                pred
            )
        })

        print(f"k={k}, seed={seed} finished")


results = pd.DataFrame(results)

results.to_csv(
    "results/k_sensitivity.csv",
    index=False
)

summary = (
    results
    .groupby(["k", "direction", "model"])["AUROC"]
    .agg(["mean", "std", "min", "max"])
)

print("\n===== K-MER SENSITIVITY =====")
print(summary)

print("\nSaved:")
print("results/k_sensitivity.csv")