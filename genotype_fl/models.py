"""Model and metrics (Model role): logistic regression trained by gradient descent."""
import numpy as np
import config


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def train_local(w, X, y, epochs=1, lr=None, l2=None):
    lr = config.LR if lr is None else lr
    l2 = config.L2 if l2 is None else l2
    w = w.copy()
    for _ in range(epochs):
        w -= lr * (X.T @ (sigmoid(X @ w) - y) / len(y) + l2 * w)
    return w


def auc(scores, labels):
    """Area under the ROC curve via ranks (scores are continuous, so ties are ignored)."""
    order = scores.argsort()
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def evaluate(w, X, y):
    s = X @ w
    return float(((s > 0) == (y == 1)).mean()), auc(s, y)


def evaluate_by_group(w, X, y, g):
    """Accuracy and AUC for each ancestry group in the test set."""
    return {int(k): evaluate(w, X[g == k], y[g == k]) for k in np.unique(g)}


def causal_recovery(w, causal):
    """Share of the planted causal SNPs that appear among the model's top-|weight| SNPs (top k = number of causal)."""
    k = len(causal)
    top = np.argsort(-np.abs(w))[:k]
    return float(len(set(top) & set(causal)) / k)
