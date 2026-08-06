
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class MetricCIs:
    mean: float
    lower: float
    upper: float


@dataclass
class EvaluationResult:
    f1: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int
    f1_ci: Optional[MetricCIs] = None
    roc_auc_ci: Optional[MetricCIs] = None
    pr_auc_ci: Optional[MetricCIs] = None

    def to_dict(self) -> Dict[str, float]:
        result = {
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }
        if self.f1_ci is not None:
            result.update(
                {
                    "f1_mean": self.f1_ci.mean,
                    "f1_lower": self.f1_ci.lower,
                    "f1_upper": self.f1_ci.upper,
                }
            )
        if self.roc_auc_ci is not None:
            result.update(
                {
                    "roc_auc_mean": self.roc_auc_ci.mean,
                    "roc_auc_lower": self.roc_auc_ci.lower,
                    "roc_auc_upper": self.roc_auc_ci.upper,
                }
            )
        if self.pr_auc_ci is not None:
            result.update(
                {
                    "pr_auc_mean": self.pr_auc_ci.mean,
                    "pr_auc_lower": self.pr_auc_ci.lower,
                    "pr_auc_upper": self.pr_auc_ci.upper,
                }
            )
        return result


def _to_numpy(x) -> np.ndarray:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import torch  # type: ignore
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
        except ImportError:
            arr = np.asarray(x)

    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _bootstrap_metric(
    metric_fn,
    y_true: np.ndarray,
    y_pred_or_score: np.ndarray,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> MetricCIs:
    rng = np.random.default_rng(seed=random_state)
    n = len(y_true)
    if n == 0:
        raise ValueError("Cannot bootstrap metrics on empty arrays.")

    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred_or_score[idx]
        try:
            samples[i] = metric_fn(yt, yp)
        except ValueError:
            samples[i] = np.nan

    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        raise ValueError("All bootstrap samples failed for this metric.")

    mean = float(np.mean(samples))
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return MetricCIs(mean=mean, lower=lower, upper=upper)


def evaluate_binary_classification(
    y_true,
    y_pred,
    y_score,
    *,
    average: str = "binary",
    pos_label: int = 1,
    bootstrap: bool = False,
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
) -> EvaluationResult:
    y_true_np = _to_numpy(y_true).astype(int)
    y_pred_np = _to_numpy(y_pred).astype(int)
    y_score_np = _to_numpy(y_score).astype(float)

    if not (len(y_true_np) == len(y_pred_np) == len(y_score_np)):
        raise ValueError("y_true, y_pred, and y_score must have the same length.")

    cm = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0

    f1 = f1_score(y_true_np, y_pred_np, average=average, pos_label=pos_label)

    roc_auc = None
    try:
        if len(np.unique(y_true_np)) == 2:
            roc_auc = roc_auc_score(y_true_np, y_score_np)
    except ValueError:
        roc_auc = None

    pr_auc = None
    try:
        if len(np.unique(y_true_np)) == 2:
            pr_auc = average_precision_score(y_true_np, y_score_np)
    except ValueError:
        pr_auc = None

    f1_ci = roc_auc_ci = pr_auc_ci = None
    if bootstrap:
        f1_ci = _bootstrap_metric(
            lambda yt, yp: f1_score(yt, yp, average=average, pos_label=pos_label),
            y_true_np,
            y_pred_np,
            n_bootstrap=n_bootstrap,
            alpha=alpha,
            random_state=random_state,
        )
        if roc_auc is not None:
            roc_auc_ci = _bootstrap_metric(
                roc_auc_score,
                y_true_np,
                y_score_np,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state,
            )
        if pr_auc is not None:
            pr_auc_ci = _bootstrap_metric(
                average_precision_score,
                y_true_np,
                y_score_np,
                n_bootstrap=n_bootstrap,
                alpha=alpha,
                random_state=random_state,
            )

    return EvaluationResult(
        f1=float(f1),
        roc_auc=float(roc_auc) if roc_auc is not None else None,
        pr_auc=float(pr_auc) if pr_auc is not None else None,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        f1_ci=f1_ci,
        roc_auc_ci=roc_auc_ci,
        pr_auc_ci=pr_auc_ci,
    )
