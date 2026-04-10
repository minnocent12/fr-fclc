"""
experiments/run_all.py
Full FR-FCLC evaluation across all methods, seeds, and settings.

Loads results from results/boolq_client_scores.json and runs:
    - 4 methods: naive, robust, fair, fr_fclc
    - 3 settings: honest, byzantine_inflate, byzantine_deflate
    - 5 random seeds for statistical reliability
    - Saves per-seed and aggregated results

Output:
    results/run_all_results.json   — full per-seed results
    results/run_all_summary.json   — mean ± std across seeds

Run:
    python -m experiments.run_all
"""

import os
import json
import numpy as np
from typing import Dict, List

from conformal.fr_fclc_pipeline import (
    run_method,
    ALPHA,
    TRIM_FRACTION,
    FAIRNESS_TOLERANCE,
    BYZANTINE_FRACTION,
)
from conformal.robust_aggregate import simulate_byzantine_scores

# ── Config ─────────────────────────────────────────────────────────────────────
SEEDS       = [42, 123, 456, 789, 1011]
METHODS     = ["naive", "robust", "fair", "fr_fclc"]
ATTACKS     = ["inflate", "deflate"]
RESULTS_DIR = "results"


# ── Load scores ────────────────────────────────────────────────────────────────
def load_scores(path: str) -> Dict[int, List[float]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Run one seed ───────────────────────────────────────────────────────────────
def run_one_seed(
    client_scores: Dict[int, List[float]],
    seed: int,
) -> Dict:
    """Run all methods × all settings for one random seed."""
    results = {}

    # Honest setting
    results["honest"] = {}
    for m in METHODS:
        results["honest"][m] = run_method(
            m, client_scores, ALPHA, TRIM_FRACTION, FAIRNESS_TOLERANCE
        )

    # Byzantine settings
    for attack in ATTACKS:
        key = f"byzantine_{attack}"
        corrupted, byz_ids = simulate_byzantine_scores(
            client_scores, BYZANTINE_FRACTION, attack, seed=seed
        )
        results[key] = {"byzantine_ids": [int(i) for i in byz_ids]}
        for m in METHODS:
            results[key][m] = run_method(
                m, corrupted, ALPHA, TRIM_FRACTION, FAIRNESS_TOLERANCE
            )

    return results


# ── Aggregate across seeds ─────────────────────────────────────────────────────
def aggregate_seeds(all_seed_results: List[Dict]) -> Dict:
    """
    Compute mean ± std of coverage_mean, coverage_gap, is_fair, meets_target
    across all seeds for each method × setting combination.
    """
    settings = list(all_seed_results[0].keys())
    summary  = {}

    for setting in settings:
        summary[setting] = {}
        for m in METHODS:
            if m not in all_seed_results[0][setting]:
                continue

            cov_means  = [r[setting][m]["coverage_mean"]  for r in all_seed_results]
            cov_gaps   = [r[setting][m]["coverage_gap"]   for r in all_seed_results]
            is_fair    = [r[setting][m]["is_fair"]        for r in all_seed_results]
            meets_tgt  = [r[setting][m]["meets_target"]   for r in all_seed_results]

            summary[setting][m] = {
                "coverage_mean_avg": float(np.mean(cov_means)),
                "coverage_mean_std": float(np.std(cov_means)),
                "coverage_gap_avg":  float(np.mean(cov_gaps)),
                "coverage_gap_std":  float(np.std(cov_gaps)),
                "fair_rate":         float(np.mean(is_fair)),
                "target_rate":       float(np.mean(meets_tgt)),
            }

    return summary


# ── Print summary table ────────────────────────────────────────────────────────
def print_summary(summary: Dict):
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (mean ± std across 5 seeds)")
    print("=" * 80)
    print(f"  {'Setting':<22} {'Method':<10} {'Cov Mean':>14} "
          f"{'Cov Gap':>12} {'Fair%':>7} {'≥90%':>7}")
    print(f"  {'-'*74}")

    for setting, setting_data in summary.items():
        for m in METHODS:
            if m not in setting_data:
                continue
            d = setting_data[m]
            print(
                f"  {setting:<22} {m:<10} "
                f"{d['coverage_mean_avg']:>6.4f}±{d['coverage_mean_std']:.4f}  "
                f"{d['coverage_gap_avg']:>6.4f}±{d['coverage_gap_std']:.4f}  "
                f"{d['fair_rate']*100:>6.1f}%  "
                f"{d['target_rate']*100:>6.1f}%"
            )
        print(f"  {'-'*74}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    scores_path = os.path.join(RESULTS_DIR, "boolq_client_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found. Run fl_simulation.py first.")
        exit(1)

    client_scores = load_scores(scores_path)
    print(f"Loaded scores for {len(client_scores)} clients")
    print(f"Running {len(SEEDS)} seeds × {len(METHODS)} methods × "
          f"{1 + len(ATTACKS)} settings...\n")

    all_seed_results = []
    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print("=" * 60)
        np.random.seed(seed)
        result = run_one_seed(client_scores, seed)
        all_seed_results.append(result)

    # Aggregate
    summary = aggregate_seeds(all_seed_results)
    print_summary(summary)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)

    full_path = os.path.join(RESULTS_DIR, "run_all_results.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(all_seed_results, f, indent=2)
    print(f"\n  Full results saved  → {full_path}")

    summary_path = os.path.join(RESULTS_DIR, "run_all_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved       → {summary_path}")

    print("\nrun_all.py — OK")