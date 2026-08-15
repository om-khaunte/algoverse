# loso sweep
import json
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


class MLP(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, seed=0, lr=1e-3, epochs=50):
    """Train a fresh MLP on the given training embeddings/labels."""
    torch.manual_seed(seed)
    mlp = MLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = mlp(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()

    return mlp


def evaluate_mlp(mlp, X_test, y_test):
    """Evaluate a trained MLP, return AUROC."""
    mlp.eval()
    with torch.no_grad():
        preds = mlp(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    return roc_auc_score(y_test, preds)


def get_embeddings_and_labels(sample_ids, df_full, X_all, id_to_index):
    """Look up precomputed embeddings and labels for a given list of sample IDs."""
    valid_ids = [sid for sid in sample_ids if sid in id_to_index]
    indices = [id_to_index[sid] for sid in valid_ids]
    X = X_all[indices]
    y = df_full.set_index("sample_id").loc[valid_ids]["label"].values
    return X, y


def run_loso_sweep(loso_folds, df_full, X_all, ids_all, seeds=(0, 1, 2)):
    """
    Runs the full leave-one-species-out sweep: for each fold, train on the
    fold's training species, test zero-shot on the held-out target species,
    repeated across multiple seeds.
    Requires: X_all/ids_all precomputed via embed.py's embed_species (run once
    on the full dataset, not per-fold, for efficiency).
    Returns a DataFrame with one row per (fold, seed).
    """
    id_to_index = {sid: i for i, sid in enumerate(ids_all)}
    results = []

    for fold in loso_folds["folds"]:
        X_train, y_train = get_embeddings_and_labels(fold["train_sample_ids"], df_full, X_all, id_to_index)
        X_test, y_test = get_embeddings_and_labels(fold["test_sample_ids"], df_full, X_all, id_to_index)

        for seed in seeds:
            mlp = train_mlp(X_train, y_train, seed=seed)
            auroc = evaluate_mlp(mlp, X_test, y_test)
            results.append({
                "fold_id": fold["fold_id"],
                "test_species": fold["test_species"],
                "seed": seed,
                "auroc": auroc,
                "n_train": len(y_train),
                "n_test": len(y_test)
            })
            print(f"Fold {fold['fold_id']} ({fold['test_species']}), seed {seed}: AUROC = {auroc:.3f}")

    return pd.DataFrame(results)


def to_registry_format(results_df, track="VF", model="ESM2_MLP"):
    """Reshapes sweep results into the shared results/registry.csv format."""
    rows = []
    for _, row in results_df.iterrows():
        rows.append({
            "run_id": f"loso_fold{row['fold_id']}_seed{row['seed']}",
            "track": track,
            "source_species": "all_except_" + row["test_species"].replace(" ", "_"),
            "target_species": row["test_species"],
            "model": model,
            "seed": row["seed"],
            "F1": "",  # ONLY AUROC WAS COMPUTED IN THIS SWEEP 
            "AUROC": row["auroc"],
            "n_train": row["n_train"],
            "n_test": row["n_test"],
            "date": datetime.date.today().isoformat()
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_full = pd.read_csv("data/vf_dataset_cdhit40_zero_shot_ready.csv")
    with open("splits/loso_folds.json") as f:
        loso_folds = json.load(f)

    # X_all / ids_all should be precomputed once via embed.py on the full dataset
    X_all = np.load("embeddings/all_sequences_X.npy")
    with open("embeddings/all_sequences_ids.txt") as f:
        ids_all = [line.strip() for line in f]

    results_df = run_loso_sweep(loso_folds, df_full, X_all, ids_all)
    results_df.to_csv("results/loso_sweep_results.csv", index=False)

    registry_df = to_registry_format(results_df)
    registry_df.to_csv("results/registry.csv", index=False)
