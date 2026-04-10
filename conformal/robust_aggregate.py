"""
conformal/robust_aggregate.py
Byzantine-robust server-side APS score aggregation using trimmed quantiles.

Naive federated CP computes the global threshold as:
    tau = quantile(all_scores, 1 - alpha)

This is vulnerable to Byzantine clients that report inflated or deflated
scores to shift the threshold in their favor. FR-FCLC replaces this with
trimmed quantile aggregation:
    1. Collect per-client score arrays
    2. Trim the top and bottom trim_fraction of scores globally
    3. Compute the (1 - alpha) quantile on the remaining scores

This provides robustness against up to trim_fraction * N_clients
Byzantine clients without sacrificing coverage for honest clients.

Scaling plan:
    Phase 4 (10 clients)  : trim_fraction=0.15 → trims 1-2 clients
    Phase 5 (100 clients) : trim_fraction=0.15 → trims 15 clients (handles 30% Byzantine)
"""

import numpy as np
from typing import Dict, List, Tuple


# ── Config ─────────────────────────────────────────────────────────────────────
ALPHA          = 0.1    # Miscoverage level — target coverage = 1 - ALPHA = 90%
TRIM_FRACTION  = 0.15   # Fraction of scores to trim from each tail


# ── Naive global quantile (baseline) ──────────────────────────────────────────
def naive_threshold(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
) -> float:
    """
    Compute naive global conformal threshold (no robustness).
    Used as the baseline to compare against trimmed quantile.

    Args:
        client_scores : dict mapping client_id -> list of APS scores
        alpha         : miscoverage level (target coverage = 1 - alpha)

    Returns:
        threshold : float in [0, 1]
    """
    all_scores = np.concatenate(list(client_scores.values())).astype(np.float32)
    n          = len(all_scores)
    q          = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold  = float(np.quantile(all_scores, q))

    print(f"  [Naive] n={n}, mean={all_scores.mean():.4f}, "
          f"std={all_scores.std():.4f}, threshold={threshold:.4f}")
    return threshold


# ── Trimmed quantile aggregation (FR-FCLC robust baseline) ────────────────────
def trimmed_threshold(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    trim_fraction: float = TRIM_FRACTION,
) -> Tuple[float, Dict]:
    """
    Compute Byzantine-robust conformal threshold using trimmed quantiles.

    Algorithm:
        1. Pool all client scores into one array
        2. Sort scores and trim the bottom and top trim_fraction
        3. Compute the (1 - alpha) quantile on trimmed scores

    This resists Byzantine clients that report extreme scores to manipulate
    the global threshold. With trim_fraction=0.15 and 100 clients, up to
    15 clients on each tail (30 total) can be fully Byzantine without
    affecting the threshold.

    Args:
        client_scores  : dict mapping client_id -> list of APS scores
        alpha          : miscoverage level
        trim_fraction  : fraction to trim from each tail (default 0.15)

    Returns:
        threshold : robust float threshold in [0, 1]
        meta      : dict with diagnostics
    """
    all_scores = np.concatenate(list(client_scores.values())).astype(np.float32)
    n          = len(all_scores)
    n_trim     = int(np.floor(n * trim_fraction))

    if n_trim == 0 or 2 * n_trim >= n:
        print(f"  [Trimmed] Warning: n={n} too small to trim safely at "
              f"fraction={trim_fraction}. Need at least {int(2/trim_fraction)+1} "
              f"scores. Falling back to naive.")
        t = naive_threshold(client_scores, alpha)
        return t, {"fallback": True, "n_total": n}

    # Sort and trim both tails
    sorted_scores  = np.sort(all_scores)
    trimmed_scores = sorted_scores[n_trim: n - n_trim]
    n_trimmed      = len(trimmed_scores)

    # Conformal quantile on trimmed scores
    q         = min(np.ceil((n_trimmed + 1) * (1 - alpha)) / n_trimmed, 1.0)
    threshold = float(np.quantile(trimmed_scores, q))

    meta = {
        "n_total":    n,
        "n_trimmed":  n_trimmed,
        "n_removed":  2 * n_trim,
        "trim_low":   float(sorted_scores[n_trim - 1]),
        "trim_high":  float(sorted_scores[n - n_trim]),
        "mean_trimmed": float(trimmed_scores.mean()),
        "std_trimmed":  float(trimmed_scores.std()),
        "threshold":  threshold,
    }

    print(f"  [Trimmed] n_total={n}, trimmed={n_trimmed} "
          f"(removed {2*n_trim} scores, {trim_fraction*100:.0f}% each tail)")
    print(f"  [Trimmed] mean={trimmed_scores.mean():.4f}, "
          f"std={trimmed_scores.std():.4f}, threshold={threshold:.4f}")

    return threshold, meta


# ── Byzantine simulation ───────────────────────────────────────────────────────
def simulate_byzantine_scores(
    client_scores: Dict[int, List[float]],
    byzantine_fraction: float = 0.30,
    attack: str = "inflate",
    seed: int = 42,
) -> Tuple[Dict[int, List[float]], List[int]]:
    """
    Simulate Byzantine clients that report corrupted APS scores.

    Attack modes:
        "inflate"  : report scores near 1.0 (raises threshold → under-coverage)
        "deflate"  : report scores near 0.0 (lowers threshold → false coverage)
        "random"   : report uniformly random scores

    Args:
        client_scores      : original honest client scores
        byzantine_fraction : fraction of clients to corrupt (default 0.30)
        attack             : type of attack ("inflate", "deflate", "random")
        seed               : random seed for reproducibility

    Returns:
        corrupted_scores : dict with Byzantine clients' scores replaced
        byzantine_ids    : list of corrupted client IDs
    """
    rng        = np.random.RandomState(seed)
    client_ids = sorted(client_scores.keys())
    n_byzantine = max(1, int(np.floor(len(client_ids) * byzantine_fraction)))

    # Select Byzantine clients randomly
    byzantine_ids = list(rng.choice(client_ids, n_byzantine, replace=False))

    corrupted = {cid: list(scores) for cid, scores in client_scores.items()}

    for cid in byzantine_ids:
        n = len(corrupted[cid])
        if attack == "inflate":
            corrupted[cid] = list(rng.uniform(0.95, 1.0, size=n))
        elif attack == "deflate":
            corrupted[cid] = list(rng.uniform(0.0, 0.05, size=n))
        elif attack == "random":
            corrupted[cid] = list(rng.uniform(0.0, 1.0, size=n))
        else:
            raise ValueError(f"Unknown attack: {attack}. Use 'inflate', 'deflate', or 'random'.")

    print(f"  [Byzantine] {n_byzantine}/{len(client_ids)} clients corrupted "
          f"(attack='{attack}', ids={byzantine_ids})")

    return corrupted, byzantine_ids


# ── Compare naive vs robust ────────────────────────────────────────────────────
def compare_aggregation(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    trim_fraction: float = TRIM_FRACTION,
    byzantine_fraction: float = 0.30,
    attack: str = "inflate",
) -> Dict:
    """
    Run a full comparison of naive vs trimmed quantile aggregation under
    Byzantine attack. Returns a summary dict for plotting and analysis.

    Args:
        client_scores      : honest per-client APS scores
        alpha              : miscoverage level
        trim_fraction      : trimming fraction for robust aggregation
        byzantine_fraction : fraction of Byzantine clients to simulate
        attack             : Byzantine attack mode

    Returns:
        results : dict with thresholds and metadata for both methods
    """
    print("\n" + "=" * 55)
    print("Aggregation Comparison: Naive vs Trimmed Quantile")
    print("=" * 55)

    # Honest scores — both methods
    print("\n--- Honest clients (no attack) ---")
    tau_naive_honest   = naive_threshold(client_scores, alpha)
    tau_robust_honest, meta_honest = trimmed_threshold(
        client_scores, alpha, trim_fraction
    )

    # Byzantine scores — both methods
    print("\n--- Byzantine attack ---")
    corrupted_scores, byz_ids = simulate_byzantine_scores(
        client_scores, byzantine_fraction, attack
    )
    tau_naive_byz   = naive_threshold(corrupted_scores, alpha)
    tau_robust_byz, meta_byz = trimmed_threshold(
        corrupted_scores, alpha, trim_fraction
    )

    results = {
        "honest": {
            "naive_threshold":   tau_naive_honest,
            "robust_threshold":  tau_robust_honest,
            "meta":              meta_honest,
        },
        "byzantine": {
            "byzantine_ids":     byz_ids,
            "attack":            attack,
            "naive_threshold":   tau_naive_byz,
            "robust_threshold":  tau_robust_byz,
            "naive_shift":       abs(tau_naive_byz - tau_naive_honest),
            "robust_shift":      abs(tau_robust_byz - tau_robust_honest),
            "meta":              meta_byz,
        },
    }

    print(f"\n--- Summary ---")
    print(f"  Honest naive threshold    : {tau_naive_honest:.4f}")
    print(f"  Honest robust threshold   : {tau_robust_honest:.4f}")
    print(f"  Byzantine naive threshold : {tau_naive_byz:.4f}  "
          f"(shift={results['byzantine']['naive_shift']:.4f})")
    print(f"  Byzantine robust threshold: {tau_robust_byz:.4f}  "
          f"(shift={results['byzantine']['robust_shift']:.4f})")
    print(f"  Robust reduction in shift : "
          f"{results['byzantine']['naive_shift'] - results['byzantine']['robust_shift']:.4f}")

    return results


# ── Main: test on saved Phase 3 results ───────────────────────────────────────
if __name__ == "__main__":
    import json, os

    scores_path = os.path.join("results", "boolq_client_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found.")
        print("Run federated/fl_simulation.py first to generate client scores.")
        exit(1)

    with open(scores_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Convert string keys to int
    client_scores = {int(k): v for k, v in raw.items()}
    print(f"Loaded scores for {len(client_scores)} clients from {scores_path}")

    results = compare_aggregation(
        client_scores,
        alpha=ALPHA,
        trim_fraction=TRIM_FRACTION,
        byzantine_fraction=0.30,
        attack="inflate",
    )

    print("\nrobust_aggregate.py — OK")