"""Federated averaging with encrypted aggregation (Crypto role), using TenSEAL (CKKS).

Key handling:
  * The hospitals hold the full context (secret key).
  * The server gets a PUBLIC copy of the context: it can add ciphertexts but cannot decrypt.
Each round: a hospital trains locally, multiplies its weights by its FedAvg weight, encrypts them
and sends the ciphertext. The server adds the ciphertexts. The hospitals decrypt the sum, which is the new global model.
Only the averaging step is encrypted. The final model can still leak information about its training data.
CKKS is approximate, so results differ from plain FedAvg by tiny rounding errors.
"""
import time
import numpy as np
import tenseal as ts
import config
from models import train_local, evaluate


def make_contexts():
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=config.HE_POLY_DEGREE,
                     coeff_mod_bit_sizes=config.HE_COEFF_BITS)
    ctx.global_scale = config.HE_SCALE
    ctx.generate_galois_keys()
    server_ctx = ctx.copy()
    server_ctx.make_context_public()                 # drops the secret key
    return ctx, server_ctx


def server_cannot_decrypt(server_ctx, ctx):
    """Demonstrate the key split: decrypting with the server's public context must fail."""
    ct = ts.ckks_vector(ctx, [1.0, 2.0, 3.0])
    server_copy = ts.ckks_vector_from(server_ctx, ct.serialize())
    try:
        server_copy.decrypt()
    except Exception:
        return True
    return False


def fedavg_he(split, n_snps, rounds=None, epochs=None, lr=None):
    rounds = rounds or config.ROUNDS
    epochs = epochs or config.LOCAL_EPOCHS
    ctx, server_ctx = make_contexts()
    w = np.zeros(n_snps, np.float32)
    history, t_enc, t_agg, t_dec, ct_bytes = [], 0.0, 0.0, 0.0, 0
    for _ in range(rounds):
        received = []
        for k, (Xh, yh) in enumerate(split["hospitals"]):
            wk = train_local(w, Xh, yh, epochs, lr)
            t = time.perf_counter()
            blob = ts.ckks_vector(ctx, (split["weights"][k] * wk).tolist()).serialize()   # hospital encrypts
            t_enc += time.perf_counter() - t
            ct_bytes = len(blob)
            received.append(ts.ckks_vector_from(server_ctx, blob))                        # server holds ciphertext only
        t = time.perf_counter()
        agg = received[0]
        for c in received[1:]:
            agg = agg + c                                                                  # server adds ciphertexts
        agg_blob = agg.serialize()
        t_agg += time.perf_counter() - t
        t = time.perf_counter()
        w = np.array(ts.ckks_vector_from(ctx, agg_blob).decrypt(), dtype=np.float32)       # hospitals decrypt
        t_dec += time.perf_counter() - t
        history.append(evaluate(w, split["Xte"], split["yte"])[1])
    n_h = len(split["hospitals"])
    return w, history, {
        "uplink_bytes_per_round": ct_bytes * n_h, "ciphertext_bytes": ct_bytes,
        "encrypt_s_per_round": t_enc / rounds, "aggregate_s_per_round": t_agg / rounds,
        "decrypt_s_per_round": t_dec / rounds,
        "server_cannot_decrypt": server_cannot_decrypt(server_ctx, ctx),
    }
