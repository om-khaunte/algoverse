import os
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# 1. Load Feature Matrices & Feature Names
# --------------------------------------------------
X_ecoli = load_npz("ecoli_shared.npz")
X_salmonella = load_npz("salmonella_shared.npz")

print("E. coli matrix:", X_ecoli.shape)
print("Salmonella matrix:", X_salmonella.shape)

# Load feature names (k-mers) for downstream CARD mapping
# Adjust filename if your k-mer list is named differently
kmer_names = pd.read_csv("shared_kmers.csv", header=None)[0].values

# --------------------------------------------------
# 2. Load IDs & Phenotypes
# --------------------------------------------------
ecoli_ids = pd.read_csv("ecoli_shared_ids.csv", header=None, names=["Genome ID"])
salmonella_ids = pd.read_csv("salmonella_shared_ids.csv", header=None, names=["Genome ID"])

ecoli_ids["Genome ID"] = ecoli_ids["Genome ID"].astype(str)
salmonella_ids["Genome ID"] = salmonella_ids["Genome ID"].astype(str)

ecoli_df = pd.read_csv(os.path.join("data", "selected", "ecolidataset.csv"))
salmonella_df = pd.read_csv(os.path.join("data", "selected", "salmonelladataset.csv"))

ecoli_df["Genome ID"] = ecoli_df["Genome ID"].astype(str)
salmonella_df["Genome ID"] = salmonella_df["Genome ID"].astype(str)

ecoli_labels = (
    ecoli_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({"Susceptible": 0, "Resistant": 1})
)

salmonella_labels = (
    salmonella_df
    .drop_duplicates("Genome ID")
    .set_index("Genome ID")["Resistant Phenotype"]
    .map({"Susceptible": 0, "Resistant": 1})
)

# Align labels strictly with matrix row order
y_ecoli = ecoli_ids["Genome ID"].map(ecoli_labels).values.astype(int)
y_salmonella = salmonella_ids["Genome ID"].map(salmonella_labels).values.astype(int)

assert not pd.isna(y_ecoli).any(), "Missing labels in E. coli dataset"
assert not pd.isna(y_salmonella).any(), "Missing labels in Salmonella dataset"

print("E. coli class distribution:", np.bincount(y_ecoli))
print("Salmonella class distribution:", np.bincount(y_salmonella))

os.makedirs("results", exist_ok=True)

# --------------------------------------------------
# 3. Label-Shuffled Negative Control Baseline
# --------------------------------------------------
print("\n=== Running Label-Shuffled Control (30 Seeds) ===")
control_results = []

for seed in range(30):
    rng = np.random.default_rng(seed)

    # Permute target vectors independently
    shuffled_ecoli = rng.permutation(y_ecoli)
    shuffled_salmonella = rng.permutation(y_salmonella)

    # Models trained on noise evaluated against target noise baseline
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", random_state=seed, n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed
        )
    }

    for name, model in models.items():
        # E. coli -> Salmonella
        model.fit(X_ecoli, shuffled_ecoli)
        pred_salm = model.predict_proba(X_salmonella)[:, 1]
        auc_fwd = roc_auc_score(shuffled_salmonella, pred_salm)
        control_results.append({
            "direction": "E_coli_to_Salmonella", "model": name, "seed": seed, "AUROC": auc_fwd
        })

        # Salmonella -> E. coli
        model.fit(X_salmonella, shuffled_salmonella)
        pred_ecoli = model.predict_proba(X_ecoli)[:, 1]
        auc_rev = roc_auc_score(shuffled_ecoli, pred_ecoli)
        control_results.append({
            "direction": "Salmonella_to_E_coli", "model": name, "seed": seed, "AUROC": auc_rev
        })

control_df = pd.DataFrame(control_results)
control_df.to_csv("results/shuffle_control.csv", index=False)

summary_control = control_df.groupby(["direction", "model"])["AUROC"].agg(["mean", "std", "min", "max"])
print("\nSHUFFLED-LABEL CONTROL SUMMARY:")
print(summary_control)

# Strict Gatekeeper Check
for (direction, model_name), row in summary_control.iterrows():
    if abs(row["mean"] - 0.50) > 0.05 or row["std"] > 0.08:
        raise RuntimeError(
            f"FAIL SAFE: Control failed for {direction} ({model_name}). "
            f"Mean: {row['mean']:.3f}, Std: {row['std']:.3f}. "
            f"Investigate pipeline leakage or phylogenetic structure before proceeding."
        )

# --------------------------------------------------
# 4. Real Zero-Shot Transfer & Feature Extraction
# --------------------------------------------------
print("\n=== Running Real Zero-Shot Experiments (3 Seeds) ===")
seeds = [42, 123, 999]
real_results = []
top_kmers_list = []

for seed in seeds:
    # --- Forward: E. coli -> Salmonella ---
    xgb_fwd = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1
    )
    xgb_fwd.fit(X_ecoli, y_ecoli)
    xgb_fwd_pred = xgb_fwd.predict_proba(X_salmonella)[:, 1]
    auc_xgb_fwd = roc_auc_score(y_salmonella, xgb_fwd_pred)

    rf_fwd = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed)
    rf_fwd.fit(X_ecoli, y_ecoli)
    rf_fwd_pred = rf_fwd.predict_proba(X_salmonella)[:, 1]
    auc_rf_fwd = roc_auc_score(y_salmonella, rf_fwd_pred)

    real_results.extend([
        {"direction": "E_coli_to_Salmonella", "model": "XGBoost", "seed": seed, "AUROC": auc_xgb_fwd},
        {"direction": "E_coli_to_Salmonella", "model": "RandomForest", "seed": seed, "AUROC": auc_rf_fwd}
    ])

    # Extract Top-500 k-mers Forward
    for model_obj, model_name in [(xgb_fwd, "XGBoost"), (rf_fwd, "RandomForest")]:
        imp = model_obj.feature_importances_
        idx = np.argsort(imp)[::-1][:500]
        top_kmers_list.append(pd.DataFrame({
            "kmer": kmer_names[idx], "importance": imp[idx],
            "model": model_name, "direction": "E_coli_to_Salmonella", "seed": seed
        }))

    # --- Reverse: Salmonella -> E. coli ---
    xgb_rev = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss", tree_method="hist",
        random_state=seed, n_jobs=-1
    )
    xgb_rev.fit(X_salmonella, y_salmonella)
    xgb_rev_pred = xgb_rev.predict_proba(X_ecoli)[:, 1]
    auc_xgb_rev = roc_auc_score(y_ecoli, xgb_rev_pred)

    rf_rev = RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=seed)
    rf_rev.fit(X_salmonella, y_salmonella)
    rf_rev_pred = rf_rev.predict_proba(X_ecoli)[:, 1]
    auc_rf_rev = roc_auc_score(y_ecoli, rf_rev_pred)

    real_results.extend([
        {"direction": "Salmonella_to_E_coli", "model": "XGBoost", "seed": seed, "AUROC": auc_xgb_rev},
        {"direction": "Salmonella_to_E_coli", "model": "RandomForest", "seed": seed, "AUROC": auc_rf_rev}
    ])

    # Extract Top-500 k-mers Reverse
    for model_obj, model_name in [(xgb_rev, "XGBoost"), (rf_rev, "RandomForest")]:
        imp = model_obj.feature_importances_
        idx = np.argsort(imp)[::-1][:500]
        top_kmers_list.append(pd.DataFrame({
            "kmer": kmer_names[idx], "importance": imp[idx],
            "model": model_name, "direction": "Salmonella_to_E_coli", "seed": seed
        }))

# Save Top-500 K-mers CSV
pd.concat(top_kmers_list, ignore_index=True).to_csv("results/top_kmers.csv", index=False)
print("Successfully saved 'results/top_kmers.csv' for Week 3 CARD mapping.")

# --------------------------------------------------
# 5. Summarize Results & Transfer Asymmetry
# --------------------------------------------------
real_df = pd.DataFrame(real_results)
real_df.to_csv("results/real_experiments.csv", index=False)

summary_real = real_df.groupby(["direction", "model"])["AUROC"].agg(["mean", "std"])
print("\nREAL ZERO-SHOT TRANSFER RESULTS:")
print(summary_real)

# Model Disagreement Check
for direction in ["E_coli_to_Salmonella", "Salmonella_to_E_coli"]:
    rf_mean = summary_real.loc[(direction, "RandomForest"), "mean"]
    xgb_mean = summary_real.loc[(direction, "XGBoost"), "mean"]
    diff = abs(rf_mean - xgb_mean)
    if diff > 0.08:
        print(f"\n[FINDING] Sharp model divergence in {direction}! RF: {rf_mean:.3f} vs XGB: {xgb_mean:.3f} (Δ={diff:.3f})")

# Transfer Asymmetry Calculation
rf_fwd = summary_real.loc[("E_coli_to_Salmonella", "RandomForest"), "mean"]
rf_rev = summary_real.loc[("Salmonella_to_E_coli", "RandomForest"), "mean"]
xgb_fwd = summary_real.loc[("E_coli_to_Salmonella", "XGBoost"), "mean"]
xgb_rev = summary_real.loc[("Salmonella_to_E_coli", "XGBoost"), "mean"]

print("\nTRANSFER ASYMMETRY (Forward - Reverse):")
print(f"  Random Forest ΔAUROC: {rf_fwd - rf_rev:+.3f}")
print(f"  XGBoost ΔAUROC:       {xgb_fwd - xgb_rev:+.3f}")