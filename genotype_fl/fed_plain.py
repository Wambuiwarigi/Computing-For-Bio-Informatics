"""Plain federated averaging (Federated role).

Each round: every hospital trains locally from the current global weights, then the server
takes the sample-weighted average. This is a hand-written loop. Flower can replace it
(a Flower client wraps `train_local`, a Flower server strategy does the averaging).
"""
import numpy as np
import config
from models import train_local, evaluate


def fedavg_plain(split, n_snps, rounds=None, epochs=None, lr=None):
    rounds = rounds or config.ROUNDS
    epochs = epochs or config.LOCAL_EPOCHS
    w = np.zeros(n_snps, np.float32)
    history = []
    for _ in range(rounds):
        local = [train_local(w, Xh, yh, epochs, lr) for Xh, yh in split["hospitals"]]
        w = sum(wi * wk for wi, wk in zip(split["weights"], local))
        history.append(evaluate(w, split["Xte"], split["yte"])[1])
    uplink = len(split["hospitals"]) * n_snps * 4        # float32 weights from every hospital
    return w, history, {"uplink_bytes_per_round": uplink}
