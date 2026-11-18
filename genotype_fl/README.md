# Privacy-preserving collaborative genotype classification

Benchmark of federated averaging (FedAvg) with homomorphic-encrypted aggregation on four simulated hospital silos.

**Research question:** Can several hospitals train one shared genotype classifier without sharing patients' DNA,
and how much accuracy and speed does each added layer of privacy cost?

## Read this first: what is real and what is simulated

- **All results here use a SIMULATED cohort**, not real 1000 Genomes data. The workspace that built this could not
  download the real files (the network blocked the AWS bucket). Results describe the **method** only. They say nothing about real disease.
- Disease labels are always planted (15 causal SNPs), as agreed by the group.
- FedAvg is a hand-written loop, **not Flower**. A Flower client would wrap `models.train_local`, and a Flower strategy would do the averaging.
- Encryption is real: TenSEAL CKKS. The server only ever holds a public context and cannot decrypt (checked in every run).
- Encryption covers **only the averaging step**. The final shared model can still leak information about its training data.
- The differential-privacy extension adds clipped, noised updates but computes **no formal epsilon**. It shows the trade-off, not a guarantee.

## Results (5 seeds, mean +/- sd; 2,500 simulated people x 3,000 SNPs; global test set of 500)

| Setup | Accuracy | AUC | Planted SNPs recovered | Data sent per round (all hospitals) |
| --- | --- | --- | --- | --- |
| Centralized (ceiling) | 0.691 +/- 0.021 | 0.763 +/- 0.017 | 0.77 | none |
| Local-only (mean of 4 hospitals) | 0.597 +/- 0.017 | 0.638 +/- 0.016 | 0.49 | none |
| FedAvg, plain | 0.693 +/- 0.023 | 0.762 +/- 0.019 | 0.77 | 48 KB |
| FedAvg + encrypted aggregation | 0.693 +/- 0.023 | 0.762 +/- 0.019 | 0.77 | 1,326 KB (about 28x more) |

- Hospitals working alone reach AUC 0.64. Federating lifts that to 0.76, matching the centralized ceiling.
- Encrypted and plain FedAvg give the same model up to CKKS rounding (max weight difference about 5e-08).
- Per round, encryption takes roughly 30 ms for the four hospitals together, adding about 2 ms and decrypting about 2 ms. One ciphertext is about 331 KB. Accuracy and AUC reproduce exactly on re-run. Timings, ciphertext sizes and the tiny weight difference vary slightly between runs because CKKS encryption is randomised.
- Per-group AUC gaps are small in this simulation (0.73 to 0.76 for the federated models). Real data may differ.
- Optional noise: AUC stays near 0.76 up to noise std 0.005, then falls to 0.74 at 0.01 and 0.69 at 0.02.

Full numbers: `results/summary.txt`. Charts: `results/figures/`.

## Files (roles from Meeting 1)

| File | Role | What it does |
| --- | --- | --- |
| `config.py` | all | Settings: seeds, sizes, rounds, CKKS and DP parameters |
| `data_prep.py` | Data | Simulated cohort, plus a loader for real PLINK 2 output |
| `simulate.py` | Data | Plants labels from causal SNPs |
| `hospitals.py` | Data | Four ancestry-based hospitals and a global test set |
| `models.py` | Model | Logistic regression, AUC, per-group metrics, causal-SNP recovery |
| `fed_plain.py` | Federated | Plain FedAvg |
| `fed_he.py` | Crypto | FedAvg with TenSEAL encrypted aggregation |
| `fed_dp.py` | Federated | Optional clipped and noised updates |
| `run_experiments.py` | everyone | Runs every setup on every seed, writes CSVs |
| `plots.py` | everyone | Makes the charts |

## Run it

```bash
pip install -r requirements.txt
python data_prep.py --selftest      # checks the real-data loader on a fake file
python run_experiments.py           # about 15 seconds for 5 seeds
python plots.py
```

## Using the real 1000 Genomes data (not run in the workspace that built this)

The steps below are untested here. Check the PLINK 2 flags against your installed version.

1. Download chr22 (`ALL.chr22.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`) and
   `integrated_call_samples_v3.20130502.ALL.panel` from the 1000 Genomes release folder (also on AWS: `s3://1000genomes`).
2. Filter to common, unlinked SNPs, then thin to about 3,000:
   ```bash
   plink2 --vcf chr22.vcf.gz --snps-only just-acgt --max-alleles 2 --maf 0.05 --geno 0.02 \
          --indep-pairwise 200 50 0.2 --out chr22_pruned
   plink2 --vcf chr22.vcf.gz --extract chr22_pruned.prune.in --thin-count 3000 --seed 1 \
          --export A --out chr22_3k
   ```
   Write down every threshold and the final SNP count. Freeze the file before any final run.
3. In `run_experiments.py`, replace `simulate_genotypes(seed)` with
   `load_plink_raw("chr22_3k.raw", "integrated_call_samples_v3.20130502.ALL.panel")`.
   Hospital A = AFR, B = EUR, C = EAS, D = SAS + AMR.
4. Check that your frequency cut-offs do not drop SNPs that matter mainly in African populations.

## Limits to state in the report

- Simulated data: no real population structure, no linkage between SNPs.
- One model family (logistic regression), no hyperparameter search, 10 rounds, 5 seeds.
- Timings are for a toy-sized problem on one machine. Do not quote them as cloud benchmarks.
