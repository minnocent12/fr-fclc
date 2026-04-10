"""
federated/fl_simulation.py
Sequential FR-FCLC federated simulation — memory-safe for Apple Silicon.

Replaces Flower's run_simulation (Ray-based) which caused OOM crashes by
attempting to load 10 × 6 GB models in parallel on a 24 GB machine.

Simulation flow per round:
    FIT phase:
        For each client sequentially:
            1. Reset LoRA adapter to current global weights
            2. Fine-tune on local train data
            3. Collect updated LoRA weights + sample count
    Aggregate:
        FedAvg all local weights → new global LoRA weights
    EVALUATE phase:
        For each client sequentially:
            1. Apply aggregated global weights
            2. Compute APS calibration scores
    Server:
        Compute naive global quantile threshold (α=0.1)

Peak memory: ~6 GB — one model, one LoRA adapter, loaded once for the
entire simulation. No Ray, no parallel actors.

Output files (same format as the original Flower version):
    results/boolq_thresholds.json
    results/boolq_client_scores.json

Run:
    python -m federated.fl_simulation
"""

import os
import json
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, List

from models.model_load import load_model_and_tokenizer
from models.lora_finetune import build_peft_model, train_one_client
from federated.fl_client import load_client_data, compute_aps_scores

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET     = "boolq"
N_CLIENTS   = 10
N_ROUNDS    = 1
ALPHA       = 0.1
SEED        = 42
RESULTS_DIR = "results"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── LoRA parameter helpers ─────────────────────────────────────────────────────
def _lora_keys(model) -> List[str]:
    return sorted(k for k in model.state_dict().keys() if "lora_" in k)


def _get_params(model, keys: List[str]) -> List[np.ndarray]:
    state = model.state_dict()
    return [state[k].detach().cpu().numpy() for k in keys]


def _set_params(model, keys: List[str], params: List[np.ndarray]) -> None:
    current = model.state_dict()
    patch   = OrderedDict(
        (k, torch.tensor(arr, dtype=current[k].dtype, device=current[k].device))
        for k, arr in zip(keys, params)
    )
    model.load_state_dict(patch, strict=False)


# ── FedAvg ─────────────────────────────────────────────────────────────────────
def _fedavg(
    all_params: List[List[np.ndarray]],
    counts: List[int],
) -> List[np.ndarray]:
    total = sum(counts)
    return [
        sum(n * p[i] for n, p in zip(counts, all_params)) / total
        for i in range(len(all_params[0]))
    ]


# ── Naive threshold ────────────────────────────────────────────────────────────
def _naive_threshold(
    client_scores: Dict[int, List[float]],
    alpha: float,
) -> float:
    all_scores = np.array(
        [s for scores in client_scores.values() for s in scores],
        dtype=np.float32,
    )
    n         = len(all_scores)
    q         = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = float(np.quantile(all_scores, q))

    print(
        f"\n  [Server] Global APS scores — "
        f"n={n}, mean={all_scores.mean():.4f}, std={all_scores.std():.4f}"
    )
    print(f"  [Server] Naive threshold (α={alpha}) : {threshold:.4f}")
    print(f"  [Server] Coverage target             : {1-alpha:.0%}")
    return threshold


# ── Save results ───────────────────────────────────────────────────────────────
def _save_results(
    thresholds: List[float],
    client_scores: Dict[int, List[float]],
) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    path = os.path.join(RESULTS_DIR, f"{DATASET}_thresholds.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"  Thresholds saved    → {path}")

    path = os.path.join(RESULTS_DIR, f"{DATASET}_client_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(client_scores, f, indent=2)
    print(f"  Client scores saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FR-FCLC Sequential Federated Simulation")
    print(f"Dataset  : {DATASET}")
    print(f"Clients  : {N_CLIENTS}")
    print(f"Rounds   : {N_ROUNDS}")
    print("=" * 60)

    # Load model once — reused across all clients and all rounds
    print("\nLoading model (once for all clients)...")
    model, tokenizer, device = load_model_and_tokenizer()
    model = build_peft_model(model, device)

    keys          = _lora_keys(model)
    global_params = _get_params(model, keys)   # initial LoRA weights
    print(f"  LoRA parameter tensors : {len(keys)}")

    round_thresholds:  List[float]             = []
    # Store only the most recent round's scores (used for conformal calibration)
    all_client_scores: Dict[int, List[float]]  = {}

    for rnd in range(N_ROUNDS):
        print(f"\n{'='*60}")
        print(f"ROUND {rnd + 1} / {N_ROUNDS}")
        print("=" * 60)

        # ── FIT phase ─────────────────────────────────────────────────────────
        print("\n--- FIT ---")
        local_params:  List[List[np.ndarray]] = []
        sample_counts: List[int]              = []

        for cid in range(N_CLIENTS):
            print(f"\n  Client {cid:03d} — fit")
            _set_params(model, keys, global_params)

            train_data = load_client_data(DATASET, cid, "train")
            metrics    = train_one_client(model, tokenizer, train_data, device=device)

            local_params.append(_get_params(model, keys))
            sample_counts.append(metrics["num_samples"])

            if device == "mps":
                torch.mps.empty_cache()

        # ── FedAvg ────────────────────────────────────────────────────────────
        print(f"\n--- FedAvg ({N_CLIENTS} clients, {sum(sample_counts)} samples) ---")
        global_params = _fedavg(local_params, sample_counts)

        # ── EVALUATE phase ────────────────────────────────────────────────────
        print("\n--- EVALUATE ---")
        round_scores: Dict[int, List[float]] = {}

        for cid in range(N_CLIENTS):
            print(f"\n  Client {cid:03d} — evaluate")
            _set_params(model, keys, global_params)

            calib_data        = load_client_data(DATASET, cid, "calibration")
            scores            = compute_aps_scores(
                model, tokenizer, calib_data, device, DATASET
            )
            round_scores[cid] = scores.tolist()

            if device == "mps":
                torch.mps.empty_cache()

        # ── Server threshold ──────────────────────────────────────────────────
        threshold = _naive_threshold(round_scores, ALPHA)
        round_thresholds.append(threshold)

        # Keep only the latest round's scores for conformal calibration
        # (most recent scores reflect the most current global model)
        all_client_scores = round_scores

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\nSaving results...")
    _save_results(round_thresholds, all_client_scores)

    print("\n" + "=" * 60)
    print("Simulation complete:")
    print(f"  Threshold history : {round_thresholds}")
    print(f"  Clients scored    : {len(all_client_scores)}")
    print("=" * 60)
    print("\nfl_simulation.py — OK")