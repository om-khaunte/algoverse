import numpy as np
import pandas as pd

from scipy.sparse import load_npz, vstack
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

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

results = []

for seed in [42, 43, 44, 45, 46]:

    base_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=seed
    )

    base_model.fit(X_ecoli, y_ecoli)

    for fraction in [0.10, 0.25, 0.50]:

        train_idx, test_idx = train_test_split(
            np.arange(len(y_shigella)),
            test_size=0.20,
            stratify=y_shigella,
            random_state=seed
        )

        n_target = int(len(train_idx) * fraction)

        rng = np.random.default_rng(seed)

        selected_idx = rng.choice(
            train_idx,
            size=n_target,
            replace=False
        )

        X_target_train = X_shigella[selected_idx]
        y_target_train = y_shigella[selected_idx]

        X_combined = vstack([
            X_ecoli,
            X_target_train
        ])

        y_combined = np.concatenate([
            y_ecoli,
            y_target_train
        ])

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

        model.fit(
            X_combined,
            y_combined
        )

        pred = model.predict_proba(
            X_shigella[test_idx]
        )[:, 1]

        auc = roc_auc_score(
            y_shigella[test_idx],
            pred
        )

        results.append({
            "direction": "E_coli_to_Shigella",
            "target_fraction": fraction,
            "seed": seed,
            "AUROC": auc
        })

        print(
            f"seed={seed} "
            f"fraction={fraction} "
            f"AUROC={auc:.4f}"
        )

results = pd.DataFrame(results)

results.to_csv(
    "results/shigella_finetuning.csv",
    index=False
)

summary = (
    results
    .groupby(["direction", "target_fraction"])["AUROC"]
    .agg(["mean", "std", "min", "max"])
)

print(summary)

print("\nSaved:")
print("results/shigella_finetuning.csv")