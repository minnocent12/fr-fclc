"""
experiments/byzantine_simulation.py
Robustness evaluation under increasing Byzantine client fractions.

Simulates Byzantine attacks at multiple corruption levels (0% to 50%)
and measures how each method's threshold and coverage gap respond.

Generates:
    plots/byzantine_robustness.png  — threshold shift vs Byzantine fraction
    plots/byzantine_gap.png         — coverage gap vs Byzantine fraction
    results/byzantine_results.json  — full numeric results

Run:
    python -m experiments.byzantine_simulation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from conformal.fr_fclc_pipeline import (
    run_method,
    ALPHA,
    TRIM_FRACTION,
    FAIRNESS_TOLERANCE,
)
from conformal.robust_aggregate import simulate_byzantine_scores

# ── Config ─────────────────────────────────────────────────────────────────────
BYZ_FRACTIONS = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
ATTACKS       = ["inflate", "deflate"]
METHODS       = ["naive", "robust", "fair", "fr_fclc"]
SEEDS         = [42, 123, 456]
RESULTS_DIR   = "results"
PLOTS_DIR     = "plots"


# ── Load scores ────────────────────────────────────────────────────────────────
def load_scores(path: str) -> Dict[int, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Run sweep ──────────────────────────────────────────────────────────────────
def run_byzantine_sweep(
    client_scores: Dict[int, List[float]],
    attack: str,
) -> Dict:
    """
    For each Byzantine fraction and seed, run all methods and collect
    coverage_mean and coverage_gap. Returns nested dict.
    """
    results = {m: {f: [] for f in BYZ_FRACTIONS} for m in METHODS}

    for byz_frac in BYZ_FRACTIONS:
        print(f"  Byzantine fraction: {byz_frac:.0%} (attack={attack})")

        for seed in SEEDS:
            if byz_frac == 0.0:
                scores_to_use = client_scores
            else:
                scores_to_use, _ = simulate_byzantine_scores(
                    client_scores, byz_frac, attack, seed=seed
                )

            for m in METHODS:
                r = run_method(
                    m, scores_to_use, ALPHA, TRIM_FRACTION, FAIRNESS_TOLERANCE
                )
                results[m][byz_frac].append({
                    "coverage_mean": r["coverage_mean"],
                    "coverage_gap":  r["coverage_gap"],
                })

    return results


# ── Aggregate sweep results ────────────────────────────────────────────────────
def aggregate_sweep(sweep: Dict) -> Dict[str, Dict]:
    """Compute mean ± std of coverage_mean and coverage_gap per method per fraction."""
    agg = {}
    for m in METHODS:
        agg[m] = {
            "fractions":        BYZ_FRACTIONS,
            "cov_mean_avg":     [],
            "cov_mean_std":     [],
            "cov_gap_avg":      [],
            "cov_gap_std":      [],
        }
        for frac in BYZ_FRACTIONS:
            vals = sweep[m][frac]
            cov_means = [v["coverage_mean"] for v in vals]
            cov_gaps  = [v["coverage_gap"]  for v in vals]
            agg[m]["cov_mean_avg"].append(float(np.mean(cov_means)))
            agg[m]["cov_mean_std"].append(float(np.std(cov_means)))
            agg[m]["cov_gap_avg"].append(float(np.mean(cov_gaps)))
            agg[m]["cov_gap_std"].append(float(np.std(cov_gaps)))
    return agg


# ── Plot robustness ────────────────────────────────────────────────────────────
def plot_robustness(agg: Dict, attack: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    colors  = {"naive": "#d62728", "robust": "#ff7f0e",
               "fair": "#1f77b4", "fr_fclc": "#2ca02c"}
    labels  = {"naive": "Naive", "robust": "Robust",
               "fair": "Fair", "fr_fclc": "FR-FCLC"}
    x       = [f * 100 for f in BYZ_FRACTIONS]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle(
        f"Robustness Under Byzantine Attack ({attack})\n"
        f"{len(SEEDS)} seeds, α={ALPHA}",
        fontsize=12
    )

    for m in METHODS:
        d   = agg[m]
        err = d["cov_mean_std"]
        ax1.plot(x, d["cov_mean_avg"], marker="o", color=colors[m], label=labels[m])
        ax1.fill_between(
            x,
            [a - e for a, e in zip(d["cov_mean_avg"], err)],
            [a + e for a, e in zip(d["cov_mean_avg"], err)],
            alpha=0.15, color=colors[m]
        )

    ax1.axhline(1 - ALPHA, color="black", linestyle="--",
                linewidth=1.2, label=f"Target ({int((1-ALPHA)*100)}%)")
    ax1.set_xlabel("Byzantine Fraction (%)")
    ax1.set_ylabel("Coverage Mean")
    ax1.set_title("Coverage Mean vs Byzantine Fraction")
    ax1.set_ylim(0.5, 1.05)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               ncol=1, fontsize=9, framealpha=0.9,
               bbox_transform=ax1.transAxes, borderaxespad=0)

    for m in METHODS:
        d   = agg[m]
        err = d["cov_gap_std"]
        ax2.plot(x, d["cov_gap_avg"], marker="o", color=colors[m], label=labels[m])
        ax2.fill_between(
            x,
            [max(0, a - e) for a, e in zip(d["cov_gap_avg"], err)],
            [a + e for a, e in zip(d["cov_gap_avg"], err)],
            alpha=0.15, color=colors[m]
        )

    ax2.axhline(FAIRNESS_TOLERANCE, color="black", linestyle="--",
                linewidth=1.2, label=f"Fairness tolerance (5%)")
    ax2.set_xlabel("Byzantine Fraction (%)")
    ax2.set_ylabel("Coverage Gap")
    ax2.set_title("Coverage Gap vs Byzantine Fraction")
    ax2.set_ylim(0, max(0.6, max(
        agg[m]["cov_gap_avg"][-1] for m in METHODS
    ) + 0.1))
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0),
               ncol=1, fontsize=9, framealpha=0.9,
               bbox_transform=ax2.transAxes, borderaxespad=0)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"byzantine_robustness_{attack}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores_path = os.path.join(RESULTS_DIR, "boolq_client_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found. Run fl_simulation.py first.")
        exit(1)

    client_scores = load_scores(scores_path)
    print(f"Loaded scores for {len(client_scores)} clients")

    all_results = {}

    for attack in ATTACKS:
        print(f"\n{'='*60}")
        print(f"Byzantine sweep — attack='{attack}'")
        print("=" * 60)

        sweep = run_byzantine_sweep(client_scores, attack)
        agg   = aggregate_sweep(sweep)
        plot_robustness(agg, attack)
        all_results[attack] = agg

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "byzantine_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Byzantine results saved → {out_path}")

    print("\nbyzantine_simulation.py — OK")