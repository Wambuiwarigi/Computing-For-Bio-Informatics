"""Hospital simulation (Data role).

Each ancestry group becomes one simulated hospital. A global test set (a share of every group)
is held out and never seen by any hospital, so every setup is judged on the same people.
"""
import numpy as np
import config


def split_hospitals(X, y, gid, seed, test_frac=None):
    test_frac = test_frac or config.TEST_FRAC
    rng = np.random.default_rng(20_000 + seed)
    test_mask = np.zeros(len(X), bool)
    for g in np.unique(gid):
        idx = np.where(gid == g)[0]
        test_mask[rng.choice(idx, int(test_frac * len(idx)), replace=False)] = True
    hospitals = []
    for g in np.unique(gid):
        m = (gid == g) & ~test_mask
        hospitals.append((X[m], y[m]))
    sizes = np.array([len(h[1]) for h in hospitals], float)
    return {
        "hospitals": hospitals,
        "weights": sizes / sizes.sum(),          # FedAvg weights: share of training samples
        "Xte": X[test_mask], "yte": y[test_mask], "gte": gid[test_mask],
    }
