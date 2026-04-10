"""
conformal/fair_threshold.py
Fairness-aware per-group conformal threshold adjustment.

Problem:
    A single global threshold tau computed from pooled APS scores will
    systematically under-cover clients whose score distributions are higher
    than average. For example, if Client A has mean APS = 0.45 and Client B
    has mean APS = 0.60, applying the same tau will give Client B lower
    empirical coverage than Client A.

Solution:
    Compute per-group thresholds that equalize empirical coverage across
    clients. Each client gets its own threshold tau_i computed from its
    local calibration scores, adjusted to achieve the target (1 - alpha)
    coverage within a fairness tolerance.

Fairness metric:
    Coverage gap = max(coverage_i) - min(coverage_i) across all clients.
    Target: coverage gap < fairness_tolerance (default 0.05 = 5%).

Scaling plan:
    Phase 4 (10 clients)  : validate per-client threshold logic
    Phase 5 (100 clients) : measure coverage gap across all clients
"""

import numpy as np
from typing import Dict, List, Tuple


# ── Config ─────────────────────────────────────────────────────────────────────
ALPHA               = 0.1    # Miscoverage level — target coverage = 90%
FAIRNESS_TOLERANCE  = 0.05   # Max acceptable coverage gap across clients


# ── Per-client threshold ───────────────────────────────────────────────────────
def per_client_threshold(
    scores: List[float],
    alpha: float = ALPHA,
) -> float:
    """
    Compute a conformal threshold for a single client using its local
    calibration scores.

    Args:
        scores : list of APS scores for this client's calibration set
        alpha  : miscoverage level

    Returns:
        threshold : float in [0, 1]
    """
    arr = np.array(scores, dtype=np.float32)
    n   = len(arr)
    if n == 0:
        return 1.0
    q         = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    threshold = float(np.quantile(arr, q))
    return threshold


# ── Fairness-aware thresholds ──────────────────────────────────────────────────
def fair_thresholds(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    fairness_tolerance: float = FAIRNESS_TOLERANCE,
    global_tau: float = None,
) -> Tuple[Dict[int, float], Dict]:
    """
    Compute per-client fairness-aware conformal thresholds.

    Strategy:
        1. Compute a per-client threshold tau_i from local calibration scores
        2. Compute the global threshold tau_global (naive pooled quantile)
        3. For each client, use max(tau_i, tau_global * adjustment) to ensure
           no client is severely under-covered relative to the global target
        4. Report coverage gap as the fairness metric

    Args:
        client_scores      : dict mapping client_id -> list of APS scores
        alpha              : miscoverage level
        fairness_tolerance : max acceptable coverage gap (default 0.05)
        global_tau         : pre-computed global threshold (optional)

    Returns:
        thresholds : dict mapping client_id -> per-client threshold
        meta       : fairness diagnostics dict
    """
    from conformal.robust_aggregate import naive_threshold

    # Global threshold as reference
    if global_tau is None:
        global_tau = naive_threshold(client_scores, alpha)

    thresholds   = {}
    local_taus   = {}
    coverages    = {}

    for cid, scores in client_scores.items():
        arr      = np.array(scores, dtype=np.float32)
        tau_i    = per_client_threshold(scores, alpha)
        local_taus[cid] = tau_i

        # Fair threshold: per-client threshold, floored at global_tau
        # This prevents any client from getting a threshold so low it
        # severely under-covers relative to the global standard
        tau_fair          = max(tau_i, global_tau)
        thresholds[cid]   = tau_fair

        # Empirical coverage: fraction of scores <= threshold
        coverages[cid] = float(np.mean(arr <= tau_fair))

    # Coverage gap
    cov_values  = list(coverages.values())
    cov_gap     = float(max(cov_values) - min(cov_values))
    cov_mean    = float(np.mean(cov_values))
    fair        = cov_gap <= fairness_tolerance

    meta = {
        "global_tau":          global_tau,
        "per_client_taus":     local_taus,
        "fair_thresholds":     thresholds,
        "per_client_coverage": coverages,
        "coverage_gap":        cov_gap,
        "coverage_mean":       cov_mean,
        "fairness_tolerance":  fairness_tolerance,
        "is_fair":             fair,
    }

    print(f"\n  [Fair] Global tau         : {global_tau:.4f}")
    print(f"  [Fair] Per-client taus    : "
          f"{', '.join(f'client_{k}={v:.4f}' for k,v in local_taus.items())}")
    print(f"  [Fair] Fair thresholds    : "
          f"{', '.join(f'client_{k}={v:.4f}' for k,v in thresholds.items())}")
    print(f"  [Fair] Per-client coverage: "
          f"{', '.join(f'client_{k}={v:.4f}' for k,v in coverages.items())}")
    print(f"  [Fair] Coverage gap       : {cov_gap:.4f} "
          f"({'FAIR' if fair else 'UNFAIR — gap exceeds tolerance'})")
    print(f"  [Fair] Coverage mean      : {cov_mean:.4f}")

    return thresholds, meta


# ── Combined robust + fair thresholds ─────────────────────────────────────────
def robust_fair_thresholds(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    trim_fraction: float = 0.15,
    fairness_tolerance: float = FAIRNESS_TOLERANCE,
) -> Tuple[Dict[int, float], Dict]:
    """
    FR-FCLC full pipeline: robust trimmed quantile aggregation +
    fairness-aware per-client threshold adjustment.

    This is the core FR-FCLC contribution:
        1. Compute global_tau using trimmed quantile (Byzantine-robust)
        2. Compute per-client thresholds using local scores
        3. Apply fairness adjustment using global_tau as the floor
        4. Report coverage gap as the fairness metric

    Args:
        client_scores      : dict mapping client_id -> list of APS scores
        alpha              : miscoverage level
        trim_fraction      : trimming fraction for robust aggregation
        fairness_tolerance : max acceptable coverage gap

    Returns:
        thresholds : dict mapping client_id -> FR-FCLC threshold
        meta       : full diagnostics dict
    """
    from conformal.robust_aggregate import trimmed_threshold

    print("\n  [FR-FCLC] Computing robust global threshold...")
    robust_tau, robust_meta = trimmed_threshold(
        client_scores, alpha, trim_fraction
    )

    print("\n  [FR-FCLC] Computing fairness-aware per-client thresholds...")
    thresholds, fair_meta = fair_thresholds(
        client_scores, alpha, fairness_tolerance, global_tau=robust_tau
    )

    meta = {
        "robust": robust_meta,
        "fair":   fair_meta,
    }

    return thresholds, meta


# ── Coverage gap analysis ──────────────────────────────────────────────────────
def coverage_gap_analysis(
    client_scores: Dict[int, List[float]],
    global_tau: float,
    fair_thresholds_dict: Dict[int, float],
) -> Dict:
    """
    Compare empirical coverage under naive global tau vs fair per-client taus.

    Args:
        client_scores       : dict mapping client_id -> APS scores
        global_tau          : single global threshold (naive)
        fair_thresholds_dict: per-client fair thresholds

    Returns:
        analysis : dict with coverage comparison
    """
    naive_coverages = {}
    fair_coverages  = {}

    for cid, scores in client_scores.items():
        arr = np.array(scores, dtype=np.float32)
        naive_coverages[cid] = float(np.mean(arr <= global_tau))
        fair_coverages[cid]  = float(np.mean(arr <= fair_thresholds_dict[cid]))

    naive_gap = max(naive_coverages.values()) - min(naive_coverages.values())
    fair_gap  = max(fair_coverages.values())  - min(fair_coverages.values())

    print(f"\n  [Coverage Analysis]")
    print(f"  {'Client':<12} {'Naive Coverage':>16} {'Fair Coverage':>14}")
    print(f"  {'-'*44}")
    for cid in sorted(client_scores.keys()):
        print(f"  client_{cid:03d}   {naive_coverages[cid]:>16.4f} "
              f"{fair_coverages[cid]:>14.4f}")
    print(f"  {'-'*44}")
    print(f"  {'Coverage gap':<12} {naive_gap:>16.4f} {fair_gap:>14.4f}")
    print(f"  Gap reduction: {naive_gap - fair_gap:.4f}")

    return {
        "naive_coverages":  naive_coverages,
        "fair_coverages":   fair_coverages,
        "naive_gap":        naive_gap,
        "fair_gap":         fair_gap,
        "gap_reduction":    naive_gap - fair_gap,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json, os

    scores_path = os.path.join("results", "boolq_client_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found.")
        print("Run federated/fl_simulation.py first.")
        exit(1)

    with open(scores_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    client_scores = {int(k): v for k, v in raw.items()}
    print(f"Loaded scores for {len(client_scores)} clients")

    print("\n" + "=" * 55)
    print("FR-FCLC: Robust + Fair Threshold Computation")
    print("=" * 55)

    # Full FR-FCLC pipeline
    thresholds, meta = robust_fair_thresholds(client_scores)

    # Coverage gap analysis
    global_tau = meta["robust"]["threshold"] if "threshold" in meta["robust"] \
                 else meta["fair"]["global_tau"]

    analysis = coverage_gap_analysis(client_scores, global_tau, thresholds)

    # Save thresholds
    os.makedirs("results", exist_ok=True)
    out = {
        "per_client_thresholds": {str(k): v for k, v in thresholds.items()},
        "meta": {
            "global_tau":     global_tau,
            "coverage_gap":   analysis["fair_gap"],
            "gap_reduction":  analysis["gap_reduction"],
        }
    }
    with open("results/fair_thresholds.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Fair thresholds saved → results/fair_thresholds.json")
    print("\nfair_threshold.py — OK")