"""
conformal/fr_fclc_pipeline.py
Full FR-FCLC end-to-end pipeline integrating:
    1. Naive baseline (global quantile, no robustness or fairness)
    2. Robust-only   (trimmed quantile, no fairness)
    3. Fair-only     (per-client threshold, no robustness)
    4. FR-FCLC full  (trimmed quantile + fairness-aware adjustment)

Each method is evaluated under:
    - Honest setting   (no Byzantine clients)
    - Byzantine attack (inflate / deflate / random)

Outputs:
    results/fr_fclc_results.json   — full metrics for all methods
    plots/coverage_comparison.png  — coverage per client per method
    plots/threshold_comparison.png — threshold values per method
    plots/coverage_gap.png         — coverage gap comparison
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from conformal.robust_aggregate import (
    naive_threshold,
    trimmed_threshold,
    simulate_byzantine_scores,
)
from conformal.fair_threshold import (
    fair_thresholds,
    robust_fair_thresholds,
    coverage_gap_analysis,
)

# ── Config ─────────────────────────────────────────────────────────────────────
ALPHA              = 0.1
TRIM_FRACTION      = 0.15
FAIRNESS_TOLERANCE = 0.05
BYZANTINE_FRACTION = 0.30
RESULTS_DIR        = "results"
PLOTS_DIR          = "plots"


# ── Empirical coverage ─────────────────────────────────────────────────────────
def empirical_coverage(
    scores: List[float],
    threshold: float,
) -> float:
    """Fraction of calibration scores at or below threshold."""
    arr = np.array(scores, dtype=np.float32)
    return float(np.mean(arr <= threshold))


def per_client_coverage(
    client_scores: Dict[int, List[float]],
    thresholds: Dict[int, float],
) -> Dict[int, float]:
    """Compute empirical coverage per client given per-client thresholds."""
    return {
        cid: empirical_coverage(client_scores[cid], thresholds[cid])
        for cid in client_scores
    }


def global_coverage(
    client_scores: Dict[int, List[float]],
    tau: float,
) -> Dict[int, float]:
    """Compute empirical coverage per client given a single global threshold."""
    return {
        cid: empirical_coverage(scores, tau)
        for cid, scores in client_scores.items()
    }


def coverage_gap(coverages: Dict[int, float]) -> float:
    """Max coverage minus min coverage across clients."""
    vals = list(coverages.values())
    return float(max(vals) - min(vals)) if len(vals) > 1 else 0.0


# ── Run one method ─────────────────────────────────────────────────────────────
def run_method(
    name: str,
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    trim_fraction: float = TRIM_FRACTION,
    fairness_tolerance: float = FAIRNESS_TOLERANCE,
) -> Dict:
    """
    Run a single calibration method and return metrics.

    Methods:
        "naive"   : global quantile, no robustness, no fairness
        "robust"  : trimmed quantile, no fairness
        "fair"    : per-client threshold using naive global tau as floor
        "fr_fclc" : trimmed quantile + fairness-aware per-client adjustment
    """
    print(f"\n  Running method: {name}")

    if name == "naive":
        tau = naive_threshold(client_scores, alpha)
        covs = global_coverage(client_scores, tau)
        thresholds = {cid: tau for cid in client_scores}

    elif name == "robust":
        tau, _ = trimmed_threshold(client_scores, alpha, trim_fraction)
        covs = global_coverage(client_scores, tau)
        thresholds = {cid: tau for cid in client_scores}

    elif name == "fair":
        naive_tau = naive_threshold(client_scores, alpha)
        thresholds, _ = fair_thresholds(
            client_scores, alpha, fairness_tolerance, global_tau=naive_tau
        )
        covs = per_client_coverage(client_scores, thresholds)

    elif name == "fr_fclc":
        thresholds, _ = robust_fair_thresholds(
            client_scores, alpha, trim_fraction, fairness_tolerance
        )
        covs = per_client_coverage(client_scores, thresholds)

    else:
        raise ValueError(f"Unknown method: {name}")

    gap  = coverage_gap(covs)
    mean = float(np.mean(list(covs.values())))

    return {
        "method":              name,
        "thresholds":          {str(k): v for k, v in thresholds.items()},
        "coverages":           {str(k): v for k, v in covs.items()},
        "coverage_mean":       mean,
        "coverage_gap":        gap,
        "is_fair":             gap <= fairness_tolerance,
        "meets_target":        mean >= (1 - alpha),
    }


# ── Full experiment ────────────────────────────────────────────────────────────
def run_full_experiment(
    client_scores: Dict[int, List[float]],
    alpha: float = ALPHA,
    trim_fraction: float = TRIM_FRACTION,
    fairness_tolerance: float = FAIRNESS_TOLERANCE,
    byzantine_fraction: float = BYZANTINE_FRACTION,
) -> Dict:
    """
    Run all four methods under honest and Byzantine settings.
    Returns a complete results dict for saving and plotting.
    """
    methods  = ["naive", "robust", "fair", "fr_fclc"]
    results  = {"honest": {}, "byzantine": {}}

    # ── Honest setting ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("HONEST SETTING (no Byzantine clients)")
    print("=" * 55)

    for m in methods:
        results["honest"][m] = run_method(
            m, client_scores, alpha, trim_fraction, fairness_tolerance
        )

    # ── Byzantine setting ──────────────────────────────────────────────────────
    for attack in ["inflate", "deflate"]:
        print(f"\n{'=' * 55}")
        print(f"BYZANTINE SETTING — attack='{attack}' "
              f"({int(byzantine_fraction*100)}% clients corrupted)")
        print("=" * 55)

        corrupted, byz_ids = simulate_byzantine_scores(
            client_scores, byzantine_fraction, attack
        )

        key = f"byzantine_{attack}"
        results[key] = {"byzantine_ids": [int(i) for i in byz_ids]}

        for m in methods:
            results[key][m] = run_method(
                m, corrupted, alpha, trim_fraction, fairness_tolerance
            )

    return results


# ── Print summary table ────────────────────────────────────────────────────────
def print_summary(results: Dict):
    """Print a clean summary table of all methods and settings."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  {'Setting':<22} {'Method':<10} {'Cov Mean':>10} "
          f"{'Cov Gap':>10} {'Fair':>6} {'≥90%':>6}")
    print(f"  {'-'*66}")

    for setting, setting_data in results.items():
        for method in ["naive", "robust", "fair", "fr_fclc"]:
            if method not in setting_data:
                continue
            m = setting_data[method]
            print(f"  {setting:<22} {method:<10} "
                  f"{m['coverage_mean']:>10.4f} "
                  f"{m['coverage_gap']:>10.4f} "
                  f"{'✓' if m['is_fair'] else '✗':>6} "
                  f"{'✓' if m['meets_target'] else '✗':>6}")
        print(f"  {'-'*66}")


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_coverage_comparison(results: Dict, n_clients: int):
    """Bar chart: per-client coverage for each method under honest setting."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    methods = ["naive", "robust", "fair", "fr_fclc"]
    labels  = ["Naive", "Robust", "Fair", "FR-FCLC"]
    colors  = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    x       = np.arange(n_clients)
    width   = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label, color) in enumerate(zip(methods, labels, colors)):
        covs = [float(v) for v in results["honest"][m]["coverages"].values()]
        ax.bar(x + i * width, covs, width, label=label, color=color, alpha=0.85)

    ax.axhline(1 - ALPHA, color="black", linestyle="--",
               linewidth=1.2, label=f"Target ({int((1-ALPHA)*100)}%)")
    ax.set_xlabel("Client ID")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Per-Client Coverage by Method (Honest Setting)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f"C{i}" for i in range(n_clients)])
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.15),
              ncol=5, framealpha=0.9, bbox_transform=ax.transAxes, borderaxespad=0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "coverage_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


def plot_coverage_gap(results: Dict):
    """Bar chart: coverage gap for each method across honest + Byzantine settings."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    methods  = ["naive", "robust", "fair", "fr_fclc"]
    labels   = ["Naive", "Robust", "Fair", "FR-FCLC"]
    colors   = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c"]
    settings = ["honest", "byzantine_inflate", "byzantine_deflate"]
    s_labels = ["Honest", "Byzantine\n(inflate)", "Byzantine\n(deflate)"]

    x     = np.arange(len(settings))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label, color) in enumerate(zip(methods, labels, colors)):
        gaps = []
        for s in settings:
            if m in results.get(s, {}):
                gaps.append(results[s][m]["coverage_gap"])
            else:
                gaps.append(0.0)
        ax.bar(x + i * width, gaps, width, label=label, color=color, alpha=0.85)

    ax.axhline(FAIRNESS_TOLERANCE, color="black", linestyle="--",
               linewidth=1.2, label=f"Fairness tolerance ({int(FAIRNESS_TOLERANCE*100)}%)")
    ax.set_xlabel("Setting")
    ax.set_ylabel("Coverage Gap")
    ax.set_title("Coverage Gap by Method and Setting")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(s_labels)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]) * 1.25)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0.0, -0.15),
              ncol=5, framealpha=0.9, bbox_transform=ax.transAxes, borderaxespad=0)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "coverage_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores_path = os.path.join(RESULTS_DIR, "boolq_client_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found.")
        print("Run federated/fl_simulation.py first.")
        exit(1)

    with open(scores_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    client_scores = {int(k): v for k, v in raw.items()}
    n_clients     = len(client_scores)
    print(f"Loaded scores for {n_clients} clients from {scores_path}")

    # Run full experiment
    results = run_full_experiment(client_scores)

    # Print summary
    print_summary(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_coverage_comparison(results, n_clients)
    plot_coverage_gap(results)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "fr_fclc_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {out_path}")

    print("\nfr_fclc_pipeline.py — OK")