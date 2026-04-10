"""
federated/fl_simulation.py
Runs the FR-FCLC federated simulation using Flower 1.27's
ClientApp / ServerApp / run_simulation API.

Simulation flow per round:
    1. Server sends global LoRA weights to selected clients
    2. Each client fine-tunes locally (fit)
    3. Each client computes calibration scores on calibration data (evaluate)
    4. Server aggregates weights (FedAvg) and scores (naive quantile)

Run:
    python -m federated.fl_simulation
"""

import os
import json
import torch
import numpy as np
import flwr as fl
from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context

from models.model_load import load_model_and_tokenizer
from federated.fl_client import FRFCLCClient
from federated.fl_server import build_strategy

# ── Config ────────────────────────────────────────────────────────────────────
DATASET = "boolq"
N_CLIENTS = 2
N_ROUNDS = 1
SEED = 42
RESULTS_DIR = "results"

np.random.seed(SEED)
torch.manual_seed(SEED)


# ── ClientApp ─────────────────────────────────────────────────────────────────
def client_fn(context: Context) -> fl.client.Client:
    """
    Create a FRFCLCClient for a given partition.

    The model is loaded once per actor instance. That actor then reuses the
    same model across fit/evaluate calls during its lifecycle.
    """
    try:
        client_id = int(context.node_config["partition-id"])
    except (KeyError, AttributeError, TypeError):
        client_id = int(context.node_id) % N_CLIENTS

    model, tokenizer, device = load_model_and_tokenizer()

    return FRFCLCClient(
        client_id=client_id,
        dataset=DATASET,
        model=model,
        tokenizer=tokenizer,
        device=device,
    ).to_client()


client_app = ClientApp(client_fn=client_fn)


# ── ServerApp ─────────────────────────────────────────────────────────────────
_strategy = build_strategy(n_clients=N_CLIENTS)


def server_fn(context: Context) -> ServerAppComponents:
    """Configure server strategy and round count."""
    config = ServerConfig(num_rounds=N_ROUNDS)
    return ServerAppComponents(strategy=_strategy, config=config)


server_app = ServerApp(server_fn=server_fn)


# ── Save results ──────────────────────────────────────────────────────────────
def save_results(strategy, dataset: str) -> None:
    """Save threshold history and client score logs."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    threshold_path = os.path.join(RESULTS_DIR, f"{dataset}_thresholds.json")
    with open(threshold_path, "w", encoding="utf-8") as file:
        json.dump(strategy.get_threshold_history(), file, indent=2)
    print(f"  Thresholds saved    → {threshold_path}")

    scores_path = os.path.join(RESULTS_DIR, f"{dataset}_client_scores.json")
    with open(scores_path, "w", encoding="utf-8") as file:
        json.dump(strategy.get_all_client_scores(), file, indent=2)
    print(f"  Client scores saved → {scores_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FR-FCLC Federated Simulation")
    print(f"Dataset  : {DATASET}")
    print(f"Clients  : {N_CLIENTS}")
    print(f"Rounds   : {N_ROUNDS}")
    print("=" * 60)

    print("\nStarting Flower simulation...")
    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=N_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 4, "num_gpus": 0.0}},
    )

    print("\nSaving results...")
    save_results(_strategy, DATASET)

    print("\n" + "=" * 60)
    print("Simulation complete:")
    print(f"  Threshold history : {_strategy.get_threshold_history()}")
    print(f"  Clients scored    : {len(_strategy.get_all_client_scores())}")
    print("=" * 60)
    print("\nfl_simulation.py — OK")