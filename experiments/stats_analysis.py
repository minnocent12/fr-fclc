"""
experiments/stats_analysis.py
Paired t-tests comparing FR-FCLC against all baselines.

Loads run_all_results.json (5 seeds × all methods × all settings)
and runs paired t-tests on:
    - coverage_mean  : higher is better
    - coverage_gap   : lower is better

Baseline comparisons:
    FR-FCLC vs Naive
    FR-FCLC vs Robust
    FR-FCLC vs Fair

Reports: t-statistic, p-value, significance at p < 0.05

Run:
    python -m experiments.stats_analysis
"""

import os
import json
import numpy as np
from scipy import stats
from typing import Dict, List

RESULTS_DIR = "results"
ALPHA_STAT  = 0.05      # Statistical significance threshold
METHODS     = ["naive", "robust", "fair"]   # Baselines to compare against FR-FCLC
SETTINGS    = ["honest", "byzantine_inflate", "byzantine_deflate"]
METRICS     = ["coverage_mean", "coverage_gap"]


# ── Load results ───────────────────────────────────────────────────────────────
def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Paired t-test ──────────────────────────────────────────────────────────────
def paired_ttest(
    fr_fclc_vals: List[float],
    baseline_vals: List[float],
    metric: str,
) -> Dict:
    """
    Paired t-test: FR-FCLC vs baseline.
    For coverage_mean: FR-FCLC should be >= baseline (higher is better).
    For coverage_gap:  FR-FCLC should be <= baseline (lower is better).
    """
    diffs = np.array(fr_fclc_vals) - np.array(baseline_vals)
    t_stat, p_val = stats.ttest_1samp(diffs, popmean=0)

    # One-sided interpretation
    if metric == "coverage_mean":
        # H1: FR-FCLC coverage_mean > baseline
        p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
        direction = "FR-FCLC > baseline"
    else:
        # H1: FR-FCLC coverage_gap < baseline
        p_one = p_val / 2 if t_stat < 0 else 1 - p_val / 2
        direction = "FR-FCLC < baseline"

    return {
        "t_stat":      float(t_stat),
        "p_two_sided": float(p_val),
        "p_one_sided": float(p_one),
        "significant": bool(p_one < ALPHA_STAT),
        "direction":   direction,
        "mean_diff":   float(np.mean(diffs)),
        "fr_fclc_mean":   float(np.mean(fr_fclc_vals)),
        "baseline_mean":  float(np.mean(baseline_vals)),
    }


# ── Run all comparisons ────────────────────────────────────────────────────────
def run_all_tests(seed_results: List[Dict]) -> Dict:
    all_tests = {}

    for setting in SETTINGS:
        all_tests[setting] = {}

        for baseline in METHODS:
            all_tests[setting][baseline] = {}

            for metric in METRICS:
                fr_vals  = [r[setting]["fr_fclc"][metric] for r in seed_results]
                bas_vals = [r[setting][baseline][metric]  for r in seed_results]

                result = paired_ttest(fr_vals, bas_vals, metric)
                all_tests[setting][baseline][metric] = result

    return all_tests


# ── Print results ──────────────────────────────────────────────────────────────
def print_results(tests: Dict):
    print("\n" + "=" * 75)
    print("PAIRED T-TEST RESULTS: FR-FCLC vs Baselines")
    print(f"Seeds: 5 | Significance threshold: p < {ALPHA_STAT}")
    print("=" * 75)

    for setting in SETTINGS:
        print(f"\n  Setting: {setting}")
        print(f"  {'Baseline':<10} {'Metric':<16} {'Mean Diff':>10} "
              f"{'t-stat':>8} {'p (1-sided)':>12} {'Sig':>5}")
        print(f"  {'-'*65}")

        for baseline in METHODS:
            for metric in METRICS:
                r   = tests[setting][baseline][metric]
                sig = "✓" if r["significant"] else "✗"
                print(
                    f"  {baseline:<10} {metric:<16} "
                    f"{r['mean_diff']:>+10.4f} "
                    f"{r['t_stat']:>8.3f} "
                    f"{r['p_one_sided']:>12.4f} "
                    f"{sig:>5}"
                )
        print()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_path = os.path.join(RESULTS_DIR, "run_all_results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run experiments/run_all.py first.")
        exit(1)

    seed_results = load_results(results_path)
    print(f"Loaded {len(seed_results)} seed results")

    tests = run_all_tests(seed_results)
    print_results(tests)

    # Save
    out_path = os.path.join(RESULTS_DIR, "stats_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2)
    print(f"  Statistical results saved → {out_path}")

    print("\nstats_analysis.py — OK")