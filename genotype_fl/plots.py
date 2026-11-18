"""Charts from the CSV results (everyone). Run after run_experiments.py."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

NAVY, TEAL, AMBER, BURNT, SLATE, STEEL = "#0F2A43", "#1C7293", "#F2A541", "#B35C00", "#55697C", "#C8D6E5"
R = config.RESULTS_DIR
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": SLATE, "axes.labelcolor": NAVY, "xtick.color": NAVY, "ytick.color": NAVY})
ORDER = ["Centralized", "Local-only (mean of 4)", "FedAvg (plain)", "FedAvg + encrypted aggregation"]
COL = {ORDER[0]: NAVY, ORDER[1]: STEEL, ORDER[2]: TEAL, ORDER[3]: BURNT}
SHORT_LABEL = {ORDER[0]: "Centralized", ORDER[1]: "Local-only", ORDER[2]: "FedAvg", ORDER[3]: "FedAvg +\nencryption"}


def collapse_local(df):
    df = df.copy()
    df["setup"] = df["setup"].where(~df["setup"].str.startswith("Local-only"), ORDER[1])
    return df


summary = collapse_local(pd.read_csv(f"{R}/summary.csv"))
by_group = collapse_local(pd.read_csv(f"{R}/by_group.csv"))
per_seed = summary.groupby(["setup", "seed"]).mean(numeric_only=True).reset_index()

# 1. AUC by setup
fig, ax = plt.subplots(figsize=(7, 4.2))
m = per_seed.groupby("setup").auc.agg(["mean", "std"]).loc[ORDER]
bars = ax.bar(range(4), m["mean"], yerr=m["std"], color=[COL[s] for s in ORDER], capsize=5, width=0.62)
for b, v in zip(bars, m["mean"]):
    ax.text(b.get_x() + b.get_width() / 2, 0.505, f"{v:.2f}", ha="center", color="white", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels([SHORT_LABEL[s] for s in ORDER])
ax.set_ylim(0.5, 0.85); ax.set_ylabel("AUC on global test set")
ax.set_title(f"Pooling is not needed: FedAvg matches centralized (mean of {per_seed.seed.nunique()} seeds)", loc="left", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/auc_by_setup.png", dpi=200); plt.close()

# 2. AUC by ancestry group
g = by_group.groupby(["setup", "group"]).auc.mean().unstack().loc[ORDER]
fig, ax = plt.subplots(figsize=(8, 4.2))
w = 0.2
for i, s in enumerate(ORDER):
    ax.bar(np.arange(4) + (i - 1.5) * w, g.loc[s].values, w, label=SHORT_LABEL[s].replace("\n", " "), color=COL[s])
ax.set_xticks(range(4)); ax.set_xticklabels([f"Hospital {h}\n({n})" for h, n in
                                            zip(config.SHORT, ["African", "European", "East Asian", "S. Asian + Amer."])], fontsize=9)
ax.set_ylim(0.5, 0.9); ax.set_ylabel("AUC (test people from that group)")
ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
ax.set_title("Per-group AUC: small gaps in this simulation, large gain from federating", loc="left", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/auc_by_group.png", dpi=200); plt.close()

# 3. Communication cost
up = per_seed.groupby("setup").uplink_bytes_per_round.mean().loc[ORDER[2:]]
fig, ax = plt.subplots(figsize=(6, 3.8))
bars = ax.bar(["FedAvg", "FedAvg +\nencryption"], up.values / 1000, color=[TEAL, BURNT], width=0.5)
ax.set_yscale("log"); ax.set_ylabel("Data sent per round (KB, log scale)")
for b, v in zip(bars, up.values / 1000):
    ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:,.0f} KB", ha="center", color=NAVY, fontweight="bold")
ax.set_ylim(10, up.max() / 1000 * 4)
ax.set_title(f"Encryption costs about {up.values[1] / up.values[0]:.0f}x more data per round", loc="left", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/communication_cost.png", dpi=200); plt.close()

# 4. DP trade-off
dp = pd.read_csv(f"{R}/dp.csv").groupby("noise_std").agg(auc=("auc", "mean"), sd=("auc", "std")).reset_index()
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.errorbar(dp.noise_std, dp.auc, yerr=dp.sd, color=TEAL, marker="o", capsize=4, linewidth=2)
cen = per_seed[per_seed.setup == "Centralized"].auc.mean()
ax.axhline(cen, color=NAVY, linestyle="--", linewidth=1); ax.text(dp.noise_std.max(), cen + 0.004, "centralized", ha="right", color=NAVY, fontsize=9)
ax.set_xlabel("Noise added to each weight update (std)"); ax.set_ylabel("AUC on global test set")
ax.set_title("More noise protects more but costs accuracy", loc="left", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/dp_tradeoff.png", dpi=200); plt.close()

# 5. Convergence
conv = pd.read_csv(f"{R}/convergence.csv")
fig, ax = plt.subplots(figsize=(6.5, 4))
for s, c, ls in [(ORDER[0], NAVY, "--"), (ORDER[2], TEAL, "-"), (ORDER[3], BURNT, ":")]:
    d = conv[conv.setup == s]
    ax.plot(d["round"], d.auc, color=c, linestyle=ls, linewidth=2.5, label=SHORT_LABEL[s].replace("\n", " "))
ax.set_xlabel("Round"); ax.set_ylabel("AUC on global test set"); ax.legend(frameon=False)
ax.set_title("Plain and encrypted FedAvg follow the same curve (seed 0)", loc="left", color=NAVY, fontsize=11)
plt.tight_layout(); plt.savefig(f"{FIG}/convergence.png", dpi=200); plt.close()
print("figures written to", FIG, sorted(os.listdir(FIG)))
