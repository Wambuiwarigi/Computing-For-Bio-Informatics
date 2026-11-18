"""Optional extension: differential-privacy style noise on client updates (Federated role).

Each hospital clips its update (new weights minus global weights) to a maximum norm and adds Gaussian noise
before sending it. We report the noise level and the accuracy it costs.
No formal epsilon is computed here (that needs a privacy accountant), so this shows the trade-off, not a privacy guarantee.
"""
import numpy as np
import config
from models import train_local, evaluate


def fedavg_dp(split, n_snps, noise_std, seed, clip=None, rounds=None, epochs=None, lr=None):
    clip = clip or config.DP_CLIP
    rounds = rounds or config.ROUNDS
    epochs = epochs or config.LOCAL_EPOCHS
    rng = np.random.default_rng(30_000 + seed)
    w = np.zeros(n_snps, np.float32)
    clipped = total = 0
    for _ in range(rounds):
        agg = np.zeros(n_snps, np.float32)
        for k, (Xh, yh) in enumerate(split["hospitals"]):
            delta = train_local(w, Xh, yh, epochs, lr) - w
            norm = float(np.linalg.norm(delta))
            if norm > clip:
                delta *= clip / norm
                clipped += 1
            total += 1
            delta = delta + rng.normal(0, noise_std, n_snps).astype(np.float32)
            agg += split["weights"][k] * delta
        w = w + agg
    return w, {"clipped_fraction": clipped / total}
