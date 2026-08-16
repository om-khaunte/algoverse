
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def finetune_model(mlp, X_finetune, y_finetune, lr=1e-3, epochs=20):
    """
    Takes an already-trained (zero-shot) MLP and continues training it
    on a small amount of target-species data.
    """
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_finetune, dtype=torch.float32)
    y_t = torch.tensor(y_finetune, dtype=torch.float32).unsqueeze(1)

    mlp.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = mlp(X_t)
        loss = loss_fn(preds, y_t)
        loss.backward()
        optimizer.step()

    return mlp


def run_finetune_recovery(zero_shot_mlp, X_target, y_target, train_size, seed=0,
                            finetune_epochs=20, evaluate_mlp_fn=None):
    """
    Splits target species data into a fine-tuning slice (train_size, e.g. 0.10)
    and a held-out remainder. Fine-tunes a copy of zero_shot_mlp on the slice,
    then evaluates both the original zero-shot model and the fine-tuned model
    on the same held-out remainder for a fair comparison.

    Returns: dict with zero_shot_auroc, finetuned_auroc, n_finetune, n_test
    """
    X_finetune, X_test_remaining, y_finetune, y_test_remaining = train_test_split(
        X_target, y_target, train_size=train_size, random_state=seed, stratify=y_target
    )

    zero_shot_auroc = evaluate_mlp_fn(zero_shot_mlp, X_test_remaining, y_test_remaining)

    import copy
    finetuned_mlp = finetune_model(copy.deepcopy(zero_shot_mlp), X_finetune, y_finetune, epochs=finetune_epochs)
    finetuned_auroc = evaluate_mlp_fn(finetuned_mlp, X_test_remaining, y_test_remaining)
