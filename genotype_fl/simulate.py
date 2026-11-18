"""Label planting (Data role).

The public genotypes carry no disease labels, so we plant them: pick a few SNPs as causal,
build a risk score from them plus noise, and call the top half "cases".
Because we know the causal SNPs, we can test whether a model found the right signal.
"""
import numpy as np
import config


def plant_labels(X, seed, n_causal=None, signal_scale=None, noise=None):
    n_causal = n_causal or config.N_CAUSAL
    signal_scale = signal_scale if signal_scale is not None else config.SIGNAL_SCALE
    noise = noise if noise is not None else config.LABEL_NOISE
    rng = np.random.default_rng(10_000 + seed)
    causal = np.sort(rng.choice(X.shape[1], n_causal, replace=False))
    beta = rng.normal(0, 1, n_causal)
    logit = X[:, causal] @ beta * signal_scale + rng.normal(0, noise, len(X))
    y = (logit > np.median(logit)).astype(np.float32)   # balanced classes
    return y, causal, beta
