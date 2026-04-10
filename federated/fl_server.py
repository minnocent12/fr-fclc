"""
federated/fl_server.py
Flower server strategy for FR-FCLC.

The server:
    1. Aggregates LoRA weights via FedAvg
    2. Collects APS scores from all clients after each round
    3. Computes a naive global quantile threshold (baseline)
    4. Stores per-client scores for later robust/fair aggregation

This file implements the BASELINE server (naive quantile, no robustness,
no fairness). The FR-FCLC robust + fair aggregation is in:
    conformal/robust_aggregate.py
    conformal/fair_threshold.py
"""

import numpy as np
import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import FitRes, EvaluateRes, Parameters
from flwr.server.client_proxy import ClientProxy

# ── Config ─────────────────────────────────────────────────────────────────────
ALPHA = 0.1    # Miscoverage level — target coverage = 1 - ALPHA = 90%


# ── Metrics aggregation helpers ────────────────────────────────────────────────
def aggregate_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics weighted by number of samples."""
    total_samples = sum(n for n, _ in metrics)
    avg_loss = sum(n * m["loss"] for n, m in metrics) / total_samples
    return {"avg_loss": avg_loss, "total_samples": total_samples}


def aggregate_eval_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Collect APS scores from all clients and compute naive global threshold.
    Returns threshold and per-client score summary.
    """
    all_scores = []
    client_summaries = {}

    for n_samples, m in metrics:
        if "aps_scores" not in m:
            continue

        # Deserialize scores from comma-separated string
        scores = np.array(
            [float(s) for s in m["aps_scores"].split(",")],
            dtype=np.float32
        )
        all_scores.extend(scores.tolist())

        cid = m.get("client_id", -1)
        client_summaries[cid] = {
            "n_samples":  n_samples,
            "mean_score": float(scores.mean()),
            "std_score":  float(scores.std()),
        }

    if not all_scores:
        return {"threshold": 1.0, "n_clients": 0}

    all_scores = np.array(all_scores)

    # Naive global quantile threshold (baseline — no robustness or fairness)
    n          = len(all_scores)
    quantile   = np.ceil((n + 1) * (1 - ALPHA)) / n
    quantile   = min(quantile, 1.0)
    threshold  = float(np.quantile(all_scores, quantile))

    print(f"\n  [Server] Global APS scores — "
          f"n={n}, mean={all_scores.mean():.4f}, "
          f"std={all_scores.std():.4f}")
    print(f"  [Server] Naive threshold (α={ALPHA}) : {threshold:.4f}")
    print(f"  [Server] Coverage target             : {1-ALPHA:.0%}")

    return {
        "threshold":  threshold,
        "n_clients":  len(client_summaries),
        "mean_score": float(all_scores.mean()),
        "std_score":  float(all_scores.std()),
    }


# ── FR-FCLC Strategy ───────────────────────────────────────────────────────────
class FRFCLCStrategy(fl.server.strategy.FedAvg):
    """
    Custom Flower strategy extending FedAvg.
    Adds APS score collection and naive global threshold computation.
    Later phases will extend this with:
        - Trimmed quantile aggregation (robustness)
        - Per-group threshold adjustment (fairness)
    """

    def __init__(self, **kwargs):
        super().__init__(
            fit_metrics_aggregation_fn=aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
            **kwargs,
        )
        self.round_thresholds: List[float] = []
        self.all_client_scores: Dict[int, List[float]] = {}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict]:
        """
        Override to extract and store per-client APS scores each round.
        """
        if failures:
            print(f"  [Server] Round {server_round} — "
                  f"{len(failures)} evaluation failures")

        # Extract scores per client before aggregating
        for _, eval_res in results:
            m = eval_res.metrics
            if "aps_scores" not in m:
                continue
            cid    = m.get("client_id", -1)
            scores = [float(s) for s in m["aps_scores"].split(",")]
            self.all_client_scores[cid] = scores

        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store threshold history
        if metrics and "threshold" in metrics:
            self.round_thresholds.append(metrics["threshold"])
            print(f"  [Server] Round {server_round} threshold stored: "
                  f"{metrics['threshold']:.4f}")

        return loss, metrics

    def get_threshold_history(self) -> List[float]:
        return self.round_thresholds

    def get_all_client_scores(self) -> Dict[int, List[float]]:
        return self.all_client_scores


# ── Strategy factory ───────────────────────────────────────────────────────────
def build_strategy(
    n_clients: int,
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 1.0,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
) -> FRFCLCStrategy:
    """
    Build and return the FR-FCLC server strategy.
    fraction_fit=1.0 means all available clients participate each round.
    """
    return FRFCLCStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min(min_fit_clients, n_clients),
        min_evaluate_clients=min(min_evaluate_clients, n_clients),
        min_available_clients=min(min_available_clients, n_clients),
    )