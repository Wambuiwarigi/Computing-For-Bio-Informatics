"""Shared settings for the federated genotype benchmark."""

SEEDS = [0, 1, 2, 3, 4]                 # each seed = new simulated cohort, labels and split
N_PER_GROUP = [800, 650, 550, 500]      # four ancestry groups -> four hospitals (uneven on purpose)
N_SNPS = 3000                           # SNPs kept (matches the ~3,000 target for chr22)
N_CAUSAL = 15                           # planted causal SNPs (agreed range: 10 to 20)
SIGNAL_SCALE = 0.9                      # strength of the planted signal
LABEL_NOISE = 1.0                       # noise added to the label logit
TEST_FRAC = 0.2                         # share of each group held out as the global test set

HOSPITALS = ["A (mostly African)", "B (mostly European)", "C (mostly East Asian)", "D (S. Asian + Americas)"]
SHORT = ["A", "B", "C", "D"]

ROUNDS = 10                             # federated rounds
LOCAL_EPOCHS = 5                        # local epochs per round
LR = 0.1
L2 = 1e-3

# CKKS (TenSEAL) settings
HE_POLY_DEGREE = 8192
HE_COEFF_BITS = [60, 40, 40, 60]
HE_SCALE = 2 ** 40

# Optional differential-privacy extension (clip + Gaussian noise on each client update)
DP_CLIP = 0.3                                     # client update norms are about 0.17 to 0.6, so this clips early rounds
DP_NOISE_STDS = [0.0, 0.002, 0.005, 0.01, 0.02]   # noise std per weight; no formal epsilon is computed

RESULTS_DIR = "results"
