import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


class MLP(nn.Module):
    def __init__(self, input_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def baseline_same_species(X, y, seed=0, lr=1e-3, epochs=50):
    """Baseline 1: train and test on the same species. Establishes the performance ceiling."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    torch.manual_seed(seed)
    model = MLP(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_train_t)
        loss = loss_fn(preds, y_train_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    return roc_auc_score(y_test, test_preds)


def baseline_majority_class(y, seed=0):
    """Baseline 2: always predict the most common label."""
    _, y_test = train_test_split(y, test_size=0.2, random_state=seed, stratify=y)
    majority_label = pd.Series(y).mode()[0]
    predictions = np.full(len(y_test), majority_label)
    try:
        return roc_auc_score(y_test, predictions)
    except ValueError:
        return float("nan")


def baseline_random(y, seed=0):
    """Baseline 3: random predictions."""
    _, y_test = train_test_split(y, test_size=0.2, random_state=seed, stratify=y)
    rng = np.random.RandomState(seed)
    random_preds = rng.rand(len(y_test))
    return roc_auc_score(y_test, random_preds)


def run_all_baselines(X, y, seed=0):
    return {
        "same_species_auroc": baseline_same_species(X, y, seed=seed),
        "majority_class_auroc": baseline_majority_class(y, seed=seed),
        "random_auroc": baseline_random(y, seed=seed),
    }
