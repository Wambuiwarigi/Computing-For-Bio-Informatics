"""
Mini prototype: federated genotype classification with encrypted aggregation.

SIMULATED data only (not real 1000 Genomes). Four hospitals split by ancestry,
planted causal SNPs, logistic regression, and four setups:
  1. Centralized         (ceiling)
  2. Local-only          (each hospital alone)
  3. FedAvg, plain
  4. FedAvg + CKKS encrypted aggregation (TenSEAL)
Flower is replaced by a hand-written FedAvg loop to keep this runnable in minutes.
"""
import time, csv
import numpy as np
import tenseal as ts

rng = np.random.default_rng(42)

# ---------------- 1. simulate genotypes (people x SNPs) ----------------
N_PER_GROUP = [650, 650, 600, 600]          # four ancestry groups -> four hospitals
M = 3000                                    # SNPs
N_CAUSAL = 15
GROUPS = ["A (mostly African)", "B (mostly European)", "C (mostly East Asian)", "D (S. Asian + Americas)"]

base_p = rng.beta(0.6, 1.4, size=M).clip(0.02, 0.98)          # shared allele frequencies
Xs, gid = [], []
for g, n in enumerate(N_PER_GROUP):
    p = (base_p + rng.normal(0, 0.06, size=M)).clip(0.01, 0.99)  # ancestry-specific drift
    Xs.append(rng.binomial(2, p, size=(n, M)).astype(np.float32))
    gid += [g] * n
X = np.vstack(Xs); gid = np.array(gid)
mu, sd = X.mean(0), X.std(0) + 1e-6
X = (X - mu) / sd

# ---------------- 2. plant labels ----------------
causal = rng.choice(M, N_CAUSAL, replace=False)
beta = rng.normal(0, 1, N_CAUSAL)
logit = X[:, causal] @ beta * 0.9 + rng.normal(0, 1.0, len(X))
y = (logit > np.median(logit)).astype(np.float32)

# ---------------- 3. split: global test set + four hospitals ----------------
test_idx = []
for g in range(4):
    idx = np.where(gid == g)[0]
    test_idx += list(rng.choice(idx, int(0.2 * len(idx)), replace=False))
test_mask = np.zeros(len(X), bool); test_mask[test_idx] = True
Xte, yte = X[test_mask], y[test_mask]
hosp = []
for g in range(4):
    m = (gid == g) & ~test_mask
    hosp.append((X[m], y[m]))
n_i = np.array([len(h[1]) for h in hosp], float); w_i = n_i / n_i.sum()

# ---------------- helpers ----------------
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

def local_train(w, Xh, yh, epochs=5, lr=0.1, l2=1e-3):
    w = w.copy()
    for _ in range(epochs):
        grad = Xh.T @ (sigmoid(Xh @ w) - yh) / len(yh) + l2 * w
        w -= lr * grad
    return w

def auc(scores, labels):
    order = scores.argsort(); ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def evaluate(w, Xe=None, ye=None):
    Xe = Xte if Xe is None else Xe; ye = yte if ye is None else ye
    s = Xe @ w
    return float(((s > 0) == (ye == 1)).mean()), float(auc(s, ye))

ROUNDS, EPOCHS = 10, 5
rows = []

# ---------------- setup 1: centralized ----------------
t0 = time.time()
Xall = np.vstack([h[0] for h in hosp]); yall = np.concatenate([h[1] for h in hosp])
w = np.zeros(M, np.float32)
for _ in range(ROUNDS * EPOCHS): w = local_train(w, Xall, yall, epochs=1)
acc, a = evaluate(w); rows.append(["Centralized", acc, a, ROUNDS, 0, 0, time.time() - t0])

# ---------------- setup 2: local-only ----------------
t0 = time.time(); accs, aucs = [], []
for Xh, yh in hosp:
    wl = np.zeros(M, np.float32)
    for _ in range(ROUNDS * EPOCHS): wl = local_train(wl, Xh, yh, epochs=1)
    ac, au = evaluate(wl); accs.append(ac); aucs.append(au)
rows.append(["Local-only (mean of 4)", np.mean(accs), np.mean(aucs), 0, 0, 0, time.time() - t0])
local_detail = list(zip(GROUPS, accs, aucs))

# ---------------- setup 3: FedAvg plain ----------------
t0 = time.time(); w = np.zeros(M, np.float32); bytes_round = M * 4 * 4  # 4 clients send float32 weights
for _ in range(ROUNDS):
    ws = [local_train(w, Xh, yh, epochs=EPOCHS) for Xh, yh in hosp]
    w = sum(wi * wk for wi, wk in zip(w_i, ws))
acc, a = evaluate(w); rows.append(["FedAvg (plain)", acc, a, ROUNDS, bytes_round, 0, time.time() - t0])
w_plain = w.copy()

# ---------------- setup 4: FedAvg + CKKS encrypted aggregation ----------------
ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2 ** 40; ctx.generate_galois_keys()          # secret key stays with the hospitals
server_ctx = ctx.copy(); server_ctx.make_context_public()          # server sees only the public context

t0 = time.time(); w = np.zeros(M, np.float32); ct_bytes = 0
for _ in range(ROUNDS):
    cts = []
    for k, (Xh, yh) in enumerate(hosp):
        wk = local_train(w, Xh, yh, epochs=EPOCHS)
        ck = ts.ckks_vector(ctx, (w_i[k] * wk).tolist())            # hospital encrypts its weighted weights
        blob = ck.serialize(); ct_bytes = len(blob)
        cts.append(ts.ckks_vector_from(server_ctx, blob))            # server receives ciphertext only
    agg = cts[0]
    for c in cts[1:]: agg = agg + c                                  # server adds ciphertexts, never decrypts
    agg_blob = agg.serialize()
    w = np.array(ts.ckks_vector_from(ctx, agg_blob).decrypt(), dtype=np.float32)  # hospitals decrypt
acc, a = evaluate(w)
rows.append(["FedAvg + encrypted aggregation", acc, a, ROUNDS, ct_bytes * 4, ct_bytes, time.time() - t0])
max_diff = float(np.abs(w - w_plain).max())

# ---------------- report ----------------
print(f"\nSimulated data: {len(X)} people x {M} SNPs, {N_CAUSAL} causal SNPs, hospital sizes {n_i.astype(int).tolist()}")
print(f"Global test set: {len(yte)} people\n")
hdr = ["Setup", "Accuracy", "AUC", "Rounds", "Bytes sent/round", "Ciphertext (bytes)", "Total time (s)"]
print(f"{hdr[0]:34s}{hdr[1]:>10s}{hdr[2]:>8s}{hdr[3]:>8s}{hdr[4]:>18s}{hdr[5]:>20s}{hdr[6]:>16s}")
for r in rows:
    print(f"{r[0]:34s}{r[1]:10.3f}{r[2]:8.3f}{r[3]:8d}{r[4]:18,d}{r[5]:20,d}{r[6]:16.2f}")
print("\nLocal-only, per hospital (tested on the global test set):")
for g, ac, au in local_detail: print(f"  {g:28s} acc {ac:.3f}  AUC {au:.3f}")
print(f"\nMax |weight difference| plain vs encrypted FedAvg: {max_diff:.2e}  (CKKS rounding only)")

with open("results.csv", "w", newline="") as f:
    cw = csv.writer(f); cw.writerow(hdr); cw.writerows(rows)