"""Run every setup on every seed and write CSV results (everyone).

Setups: Centralized, Local-only, FedAvg (plain), FedAvg + encrypted aggregation, and the optional DP extension.
All setups see the same cohort, the same labels and the same global test set within a seed.
Outputs (in results/): summary.csv, by_group.csv, convergence.csv, he_costs.csv, dp.csv, summary.txt
"""
import os, time, argparse
import numpy as np
import pandas as pd
import config
from data_prep import simulate_genotypes
from simulate import plant_labels
from hospitals import split_hospitals
from models import train_local, evaluate, evaluate_by_group, causal_recovery
from fed_plain import fedavg_plain
from fed_he import fedavg_he
from fed_dp import fedavg_dp


def run_seed(seed):
    X, gid = simulate_genotypes(seed)
    y, causal, _ = plant_labels(X, seed)
    split = split_hospitals(X, y, gid, seed)
    m = X.shape[1]
    Xte, yte, gte = split["Xte"], split["yte"], split["gte"]
    steps = config.ROUNDS * config.LOCAL_EPOCHS
    S, G, C, HE, DP = [], [], [], [], []

    def log(setup, w, rounds, uplink, ct, secs):
        acc, au = evaluate(w, Xte, yte)
        S.append(dict(seed=seed, setup=setup, accuracy=acc, auc=au, causal_recovery=causal_recovery(w, causal),
                      rounds=rounds, uplink_bytes_per_round=uplink, ciphertext_bytes=ct, seconds=secs))
        for g, (a, u) in evaluate_by_group(w, Xte, yte, gte).items():
            G.append(dict(seed=seed, setup=setup, group=config.SHORT[g], accuracy=a, auc=u))

    # 1. Centralized (ceiling): all hospitals' training data pooled
    t = time.perf_counter()
    Xall = np.vstack([h[0] for h in split["hospitals"]]); yall = np.concatenate([h[1] for h in split["hospitals"]])
    w, hist = np.zeros(m, np.float32), []
    for _ in range(config.ROUNDS):
        w = train_local(w, Xall, yall, config.LOCAL_EPOCHS)
        hist.append(evaluate(w, Xte, yte)[1])
    log("Centralized", w, 0, 0, 0, time.perf_counter() - t)
    if seed == config.SEEDS[0]:
        C += [dict(setup="Centralized", round=i + 1, auc=a) for i, a in enumerate(hist)]

    # 2. Local-only: each hospital trains alone for the same number of steps
    for k, (Xh, yh) in enumerate(split["hospitals"]):
        t = time.perf_counter()
        w = train_local(np.zeros(m, np.float32), Xh, yh, steps)
        log(f"Local-only {config.SHORT[k]}", w, 0, 0, 0, time.perf_counter() - t)

    # 3. FedAvg, plain
    t = time.perf_counter()
    w_plain, hist, st = fedavg_plain(split, m)
    log("FedAvg (plain)", w_plain, config.ROUNDS, st["uplink_bytes_per_round"], 0, time.perf_counter() - t)
    if seed == config.SEEDS[0]:
        C += [dict(setup="FedAvg (plain)", round=i + 1, auc=a) for i, a in enumerate(hist)]

    # 4. FedAvg + encrypted aggregation
    t = time.perf_counter()
    w_he, hist, st = fedavg_he(split, m)
    log("FedAvg + encrypted aggregation", w_he, config.ROUNDS, st["uplink_bytes_per_round"], st["ciphertext_bytes"],
        time.perf_counter() - t)
    HE.append(dict(seed=seed, max_weight_diff_vs_plain=float(np.abs(w_he - w_plain).max()),
                   **{k: v for k, v in st.items() if k.endswith("_per_round") and k != "uplink_bytes_per_round"},
                   server_cannot_decrypt=st["server_cannot_decrypt"]))
    if seed == config.SEEDS[0]:
        C += [dict(setup="FedAvg + encrypted aggregation", round=i + 1, auc=a) for i, a in enumerate(hist)]

    # 5. Optional: DP-style noise on client updates (plain aggregation)
    for sd in config.DP_NOISE_STDS:
        w, st = fedavg_dp(split, m, sd, seed)
        acc, au = evaluate(w, Xte, yte)
        DP.append(dict(seed=seed, noise_std=sd, accuracy=acc, auc=au, causal_recovery=causal_recovery(w, causal),
                       clipped_fraction=st["clipped_fraction"]))
    return S, G, C, HE, DP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=len(config.SEEDS), help="number of seeds to run")
    a = ap.parse_args()
    seeds = config.SEEDS[:a.seeds]
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    S, G, C, HE, DP = [], [], [], [], []
    for sd in seeds:
        t = time.perf_counter()
        s, g, c, h, d = run_seed(sd)
        S += s; G += g; C += c; HE += h; DP += d
        print(f"seed {sd} done in {time.perf_counter() - t:.1f}s")
    out = lambda n: os.path.join(config.RESULTS_DIR, n)
    summary = pd.DataFrame(S); by_group = pd.DataFrame(G)
    summary.to_csv(out("summary.csv"), index=False); by_group.to_csv(out("by_group.csv"), index=False)
    pd.DataFrame(C).to_csv(out("convergence.csv"), index=False)
    pd.DataFrame(HE).to_csv(out("he_costs.csv"), index=False); pd.DataFrame(DP).to_csv(out("dp.csv"), index=False)

    # headline table: local-only rows are averaged over the four hospitals first, then over seeds
    summary["setup_g"] = summary["setup"].where(~summary["setup"].str.startswith("Local-only"), "Local-only (mean of 4)")
    per_seed = summary.groupby(["setup_g", "seed"]).mean(numeric_only=True).reset_index()
    order = ["Centralized", "Local-only (mean of 4)", "FedAvg (plain)", "FedAvg + encrypted aggregation"]
    lines = [f"Seeds: {len(seeds)}   Cohort: {sum(config.N_PER_GROUP)} simulated people x {config.N_SNPS} SNPs, "
             f"{config.N_CAUSAL} planted causal SNPs\n"]
    lines.append(f"{'Setup':34s}{'Accuracy':>16s}{'AUC':>16s}{'Causal recovery':>18s}{'Uplink B/round':>16s}{'Seconds':>10s}")
    for s in order:
        d = per_seed[per_seed.setup_g == s]
        f = lambda c: f"{d[c].mean():.3f} +/- {d[c].std(ddof=1):.3f}"
        lines.append(f"{s:34s}{f('accuracy'):>16s}{f('auc'):>16s}{f('causal_recovery'):>18s}"
                     f"{d['uplink_bytes_per_round'].mean():>16,.0f}{d['seconds'].mean():>10.2f}")
    he = pd.DataFrame(HE)
    lines.append(f"\nEncrypted aggregation: max |weight difference| vs plain FedAvg = {he.max_weight_diff_vs_plain.max():.2e}; "
                 f"server could NOT decrypt in {int(he.server_cannot_decrypt.sum())}/{len(he)} seeds")
    lines.append(f"Per round (mean over seeds): encrypt {he.encrypt_s_per_round.mean() * 1000:.0f} ms, "
                 f"aggregate {he.aggregate_s_per_round.mean() * 1000:.1f} ms, decrypt {he.decrypt_s_per_round.mean() * 1000:.1f} ms; "
                 f"one ciphertext = {int(summary[summary.setup == 'FedAvg + encrypted aggregation'].ciphertext_bytes.iloc[0]):,} bytes")
    gg = by_group.copy()
    gg["setup"] = gg["setup"].where(~gg["setup"].str.startswith("Local-only"), "Local-only (mean of 4)")
    tab = gg.groupby(["setup", "group"]).auc.mean().unstack().loc[order]
    lines.append("\nAUC by ancestry group on the global test set (mean over seeds):\n" + tab.round(3).to_string())
    dp = pd.DataFrame(DP).groupby("noise_std")[["accuracy", "auc", "causal_recovery", "clipped_fraction"]].mean()
    lines.append("\nOptional DP-style noise on client updates (mean over seeds; no formal epsilon):\n" + dp.round(3).to_string())
    text = "\n".join(lines)
    open(out("summary.txt"), "w").write(text + "\n")
    print("\n" + text)


if __name__ == "__main__":
    main()
