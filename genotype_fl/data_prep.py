"""Data preparation (Data role).

Two ways to get a genotype matrix X (people x SNPs, standardised) plus an ancestry group id per person:

  simulate_genotypes()  : synthetic cohort with four ancestry groups. Used for every result in this repo.
  load_plink_raw()      : reads a PLINK 2 `--export A` .raw file plus the 1000 Genomes panel file.
                          Use this once you have the real filtered chr22 data (see README).

Run `python data_prep.py --selftest` to check the real-data loader on a fake file in the same format.
"""
import argparse, os, tempfile
import numpy as np
import pandas as pd
import config


def standardise(X):
    X = X.astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd


def simulate_genotypes(seed, n_per_group=None, n_snps=None):
    """Four ancestry groups share base allele frequencies but each drifts a little (a crude stand-in for ancestry)."""
    n_per_group = n_per_group or config.N_PER_GROUP
    n_snps = n_snps or config.N_SNPS
    rng = np.random.default_rng(seed)
    base_p = rng.beta(0.6, 1.4, size=n_snps).clip(0.02, 0.98)
    Xs, gid = [], []
    for g, n in enumerate(n_per_group):
        p = (base_p + rng.normal(0, 0.06, size=n_snps)).clip(0.01, 0.99)
        Xs.append(rng.binomial(2, p, size=(n, n_snps)))
        gid += [g] * n
    return standardise(np.vstack(Xs)), np.array(gid)


# 1000 Genomes super-population -> hospital index (A, B, C, D)
SUPERPOP_TO_HOSPITAL = {"AFR": 0, "EUR": 1, "EAS": 2, "SAS": 3, "AMR": 3}


def load_plink_raw(raw_path, panel_path):
    """Load `plink2 --export A` output. Returns X (standardised), hospital id per person, sample ids.
    Missing calls (NA) are replaced by the SNP mean before standardising."""
    raw = pd.read_csv(raw_path, sep=r"\s+")
    panel = pd.read_csv(panel_path, sep=r"\s+")
    snp_cols = [c for c in raw.columns if c not in ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")]
    geno = raw[snp_cols].to_numpy(dtype=np.float32)
    col_mean = np.nanmean(geno, axis=0)
    idx = np.where(np.isnan(geno))
    geno[idx] = np.take(col_mean, idx[1])
    sp = raw["IID"].map(panel.set_index("sample")["super_pop"])
    keep = sp.map(SUPERPOP_TO_HOSPITAL).notna().to_numpy()
    gid = sp[keep].map(SUPERPOP_TO_HOSPITAL).to_numpy(dtype=int)
    return standardise(geno[keep]), gid, raw.loc[keep, "IID"].to_numpy()


def _selftest():
    rng = np.random.default_rng(0)
    n, m = 60, 25
    pops = rng.choice(["AFR", "EUR", "EAS", "SAS", "AMR", "XXX"], n)
    ids = [f"HG{i:05d}" for i in range(n)]
    geno = rng.integers(0, 3, (n, m)).astype(float)
    geno[rng.random((n, m)) < 0.02] = np.nan
    with tempfile.TemporaryDirectory() as d:
        raw = pd.DataFrame({"FID": ids, "IID": ids, "PAT": 0, "MAT": 0, "SEX": 0, "PHENOTYPE": -9})
        raw = pd.concat([raw, pd.DataFrame(geno, columns=[f"rs{i}_A" for i in range(m)])], axis=1)
        raw.to_csv(os.path.join(d, "fake.raw"), sep="\t", index=False, na_rep="NA")
        pd.DataFrame({"sample": ids, "pop": "XXX", "super_pop": pops, "gender": "female"}).to_csv(
            os.path.join(d, "panel.txt"), sep="\t", index=False)
        X, gid, iid = load_plink_raw(os.path.join(d, "fake.raw"), os.path.join(d, "panel.txt"))
    expected = int((pops != "XXX").sum())
    assert X.shape == (expected, m), X.shape
    assert not np.isnan(X).any()
    assert set(np.unique(gid)) <= {0, 1, 2, 3}
    assert abs(float(X.mean())) < 1e-5
    print(f"selftest OK: loader parsed {X.shape[0]} people x {X.shape[1]} SNPs, "
          f"dropped {n - expected} people with unmapped populations, no NaNs left")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        X, g = simulate_genotypes(0)
        print("simulated cohort:", X.shape, "group sizes:", np.bincount(g).tolist())
