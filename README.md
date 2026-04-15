# FR-FCLC: Fair and Robust Federated Conformal Logit Calibration

> Mitigating Hallucinations in Federated Large Language Models via Fair and Robust Conformal Prediction

**Course:** Distributed AI — Spring 2026  
**Author:** Mirenge Innocent  
**Status:** Complete — 100-client federated simulation on Colab GPU, full evaluation across 5 seeds, Byzantine robustness sweep, and statistical significance tests complete

---

## Overview

FR-FCLC is a post-hoc calibration framework for federated LLMs. It computes
Adaptive Prediction Set (APS) scores locally on each federated client and
aggregates them at the server to produce a global conformal threshold with
statistical coverage guarantees. Byzantine-robust trimmed quantile aggregation
and fairness-aware per-group threshold adjustment are the core contributions,
validated across 100 clients, 3 settings, and 5 random seeds on GPU (Colab).

---

## Hardware Requirements

**Primary (large-scale K=100 experiments):**

| Component | Specification |
|---|---|
| Platform | Google Colab Pro |
| GPU | NVIDIA RTX PRO 6000 Blackwell / A100-SXM4-80GB |
| Backend | CUDA |
| Runtime | ~23 min for 100-client simulation |

**Ablation (K=10 experiments):**

| Component | Specification |
|---|---|
| Device | MacBook Pro 14-inch, November 2024 |
| Chip | Apple M4 Pro |
| Memory | 24 GB Unified Memory |
| Backend | MPS (Metal Performance Shaders) |

---

## Project Structure

```
fr-fclc/
├── data/
│   ├── preprocess.py          # Download and preprocess BoolQ + TruthfulQA
│   ├── client_splits.py       # Dirichlet non-IID partitioning across clients
│   ├── __init__.py
│   └── processed/             # Generated data (gitignored)
│       ├── boolq_train.json
│       ├── boolq_calibration.json
│       ├── boolq_test.json
│       ├── truthfulqa_*.json
│       └── clients/
│           ├── boolq/
│           │   ├── client_000/
│           │   │   ├── train.json
│           │   │   └── calibration.json
│           │   └── ...
│           └── truthfulqa/
│               └── ...
├── models/
│   ├── model_load.py          # Load Qwen2.5-3B-Instruct with 4-bit quantization
│   ├── lora_finetune.py       # LoRA adapter setup and per-client training loop
│   ├── client_ft_test.py      # Single-client fine-tuning test
│   └── __init__.py
├── federated/
│   ├── fl_client.py           # Flower NumPyClient: local training + APS scoring
│   ├── fl_server.py           # Custom FedAvg strategy with score aggregation
│   ├── fl_simulation.py       # Sequential load-once simulation (memory-safe)
│   └── __init__.py
├── conformal/                 # Phase 4: robust + fair aggregation (complete)
│   ├── aps_scores.py
│   ├── server_aggregate_naive.py
│   ├── robust_aggregate.py    # Trimmed quantile + Byzantine simulation
│   ├── fair_threshold.py      # Per-client fairness-aware threshold adjustment
│   ├── fr_fclc_pipeline.py    # Full integrated pipeline: all 4 methods × 3 settings
│   └── __init__.py
├── experiments/               # Phase 5: full evaluation (complete)
│   ├── run_all.py             # 4 methods × 3 settings × 5 seeds
│   ├── byzantine_simulation.py # Robustness sweep: 0%–50% corruption
│   ├── stats_analysis.py      # Paired t-tests: FR-FCLC vs baselines
│   └── __init__.py
├── plots/                     # Generated figures
│   ├── boolq_client_distribution.png
│   ├── coverage_comparison.png        # Per-client coverage: all 4 methods (honest)
│   ├── coverage_gap.png               # Coverage gap: all methods × all settings
│   ├── byzantine_robustness_inflate.png  # Coverage vs Byzantine fraction (inflate)
│   └── byzantine_robustness_deflate.png  # Coverage vs Byzantine fraction (deflate)
├── results/                   # Simulation outputs (JSON)
│   ├── boolq_thresholds.json          # Naive global threshold per round
│   ├── boolq_client_scores.json       # Per-client APS scores from simulation
│   ├── fair_thresholds.json           # Per-client FR-FCLC thresholds
│   ├── fr_fclc_results.json           # Full pipeline results: all methods × settings
│   ├── run_all_results.json           # Per-seed results: 5 seeds × 4 methods × 3 settings
│   ├── run_all_summary.json           # Aggregated mean ± std across seeds
│   ├── byzantine_results.json         # Robustness sweep: 0%–50% corruption
│   └── stats_analysis.json            # Paired t-test results
├── debug_tokenizer.py         # Tokenizer diagnostic script
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/minnocent12/fr-fclc.git
cd fr-fclc
```

### 2. Create and activate conda environment

```bash
conda create -n fr-fclc python=3.11 -y
conda activate fr-fclc
```

### 3. Install PyTorch (Apple Silicon MPS)

```bash
pip install torch torchvision torchaudio
```

Verify MPS is available:
```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# Expected: MPS available: True
```

### 4. Install all project dependencies

```bash
pip install transformers accelerate peft datasets
pip install bitsandbytes --prefer-binary
pip install "flwr[simulation]"
pip install scipy scikit-learn matplotlib seaborn pandas
```

### 5. Set PYTHONPATH

```bash
conda env config vars set PYTHONPATH=.
conda activate fr-fclc
```

---

## Verified Package Versions

| Package | Version |
|---|---|
| torch | 2.11.0 |
| transformers | 5.3.0 |
| peft | 0.18.1 |
| datasets | 4.8.4 |
| flwr | 1.27.0 |
| bitsandbytes | 0.49.2 |
| scipy | 1.17.1 |

---

## Reproducing Results — Phase by Phase

### Phase 1: Data Preparation

**Step 1 — Preprocess datasets**
```bash
python data/preprocess.py
```
Downloads BoolQ and TruthfulQA from Hugging Face and saves 6 JSON files
to `data/processed/` (train, calibration, test for each dataset).

**Step 2 — Generate non-IID client splits**
```bash
python data/client_splits.py
```
Partitions training data across clients using Dirichlet (α = 1.0).
BoolQ uses label-based Dirichlet split. TruthfulQA uses random split
(targets are near-unique, so label-based Dirichlet is not meaningful).

**Config options in `client_splits.py`:**

| Parameter | Current Value | Description |
|---|---|---|
| `N_CLIENTS` | 100 | Number of simulated clients |
| `ALPHA` | 1.0 | Dirichlet concentration (lower = more heterogeneous) |
| `MIN_SAMPLES` | 10 | Warn if any client has fewer samples |
| `MERGE_TINY` | False | Merge tiny clients into largest (use with α=0.5) |

---

### Phase 2: Model Loading and LoRA Fine-Tuning

**Step 3 — Verify model loads correctly**
```bash
python -m models.model_load
```
Loads Qwen2.5-3B-Instruct with 4-bit NF4 quantization on MPS.
Runs a quick inference test and prints model stats.

Expected output:
```
Parameters     : 1.70B
Loading mode   : 4bit
Device         : mps
MPS available  : True
Response : Yes, the sky appears blue due to Rayleigh scattering...
```

**Step 4 — Test single-client LoRA fine-tuning**
```bash
python -m models.lora_finetune
```
Applies LoRA (r=8) to the quantized model and runs one training epoch
on client_000 data. Saves adapter to `results/lora_client_000/`.

Expected output:
```
Trainable params : 3,686,400  (0.217% of total)
Avg loss    : ~2.07
LoRA adapter saved → results/lora_client_000
```

---

### Phase 3: Federated Simulation and APS Calibration

**Step 5 — Run federated simulation**
```bash
python -m federated.fl_simulation
```

> **Memory note:** The simulation uses a sequential load-once approach —
> the model is loaded once (~6 GB) and LoRA weights are reset per client.
> This replaces the original Flower/Ray `run_simulation` which attempted
> 10 × 6 GB in parallel and OOM-crashed on 24 GB M4 Pro.

**Config options in `fl_simulation.py`:**

| Parameter | Current Value | Notes |
|---|---|---|
| `N_CLIENTS` | 10 | Patched to 100 in Colab via `fr_fclc_colab.ipynb` |
| `N_ROUNDS` | 1 | Single federated round |
| `DATASET` | "boolq" | Switch to "truthfulqa" when validated |

**Runtime:** ~23 min on RTX PRO 6000 Blackwell (100 clients, 1 epoch each).

Expected output after 1 round (100 clients):
```
[Server] Global APS scores — n=~19000, mean=~0.455, std=~0.26
[Server] Naive threshold (α=0.1) : ~0.776
[Server] Coverage target         : 90%
```

Results saved to:
```
results/boolq_thresholds.json
results/boolq_client_scores.json
```

---

### Phase 4: Robust & Fair Aggregation (FR-FCLC Core)

**Step 6 — Byzantine-robust threshold (trimmed quantile)**
```bash
python -m conformal.robust_aggregate
```

**Step 7 — Fairness-aware per-client thresholds**
```bash
python -m conformal.fair_threshold
```

**Step 8 — Full FR-FCLC pipeline**
```bash
python -m conformal.fr_fclc_pipeline
```

Runs all 4 methods across 3 settings and generates plots.

**Methods compared:**

| Method | Robustness | Fairness |
|---|---|---|
| Naive | ✗ | ✗ |
| Robust | ✅ Trimmed quantile | ✗ |
| Fair | ✗ | ✅ Per-client threshold |
| FR-FCLC | ✅ Trimmed quantile | ✅ Per-client threshold |

**Phase 4 results (100 clients, 1 round):**

| Setting | Method | Cov Mean | Cov Gap | Fair | ≥90% |
|---|---|---|---|---|---|
| Honest | Naive | 0.9010 | 0.2632 | ✗ | ✓ |
| Honest | Robust | 0.7811 | 0.3684 | ✗ | ✗ |
| Honest | Fair | 0.9477 | 0.0947 | ✗ | ✓ |
| Honest | **FR-FCLC** | **0.9432** | **0.0421** | **✓** | **✓** |
| Byzantine (inflate) | Naive | 0.9107 | 0.4737 | ✗ | ✓ |
| Byzantine (inflate) | **FR-FCLC** | **0.9644** | **0.0946** | ✗ | ✓ |
| Byzantine (deflate) | Naive | 0.8972 | 0.2947 | ✗ | ✗ |
| Byzantine (deflate) | **FR-FCLC** | **0.9605** | **0.0947** | ✗ | ✓ |

Outputs saved to:
```
results/fr_fclc_results.json
plots/coverage_comparison.png
plots/coverage_gap.png
```

---

### Phase 5: Full Evaluation

**Step 9 — Multi-seed evaluation**
```bash
python -m experiments.run_all
```
Runs 4 methods × 3 settings × 5 random seeds. Outputs aggregated
mean ± std summary table and saves per-seed results.

**Step 10 — Byzantine robustness sweep**
```bash
python -m experiments.byzantine_simulation
```
Sweeps Byzantine fraction from 0% to 50% for both inflate and deflate
attacks across 3 seeds. Generates two robustness plots.

**Step 11 — Statistical significance tests**
```bash
python -m experiments.stats_analysis
```
Paired t-tests comparing FR-FCLC against each baseline on coverage_mean
and coverage_gap across all settings. Requires `run_all.py` to run first.

**Phase 5 results summary (mean ± std across 5 seeds, K=100 clients):**

| Setting | Method | Cov Mean | Cov Gap | Fair% | ≥90% |
|---|---|---|---|---|---|
| Honest | Naive | 0.9010 ± 0.0000 | 0.2632 ± 0.0000 | 0% | 100% |
| Honest | Robust | 0.7811 ± 0.0000 | 0.3684 ± 0.0000 | 0% | 0% |
| Honest | Fair | 0.9477 ± 0.0000 | 0.0947 ± 0.0000 | 0% | 100% |
| Honest | **FR-FCLC** | **0.9432 ± 0.0000** | **0.0421 ± 0.0000** | **100%** | **100%** |
| Byzantine inflate | Naive | 0.9107 ± 0.0088 | 0.4737 ± 0.0577 | 0% | 100% |
| Byzantine inflate | Robust | 0.7969 ± 0.0174 | 0.7895 ± 0.0577 | 0% | 0% |
| Byzantine inflate | Fair | 0.9700 ± 0.0013 | 0.0946 ± 0.0002 | 0% | 100% |
| Byzantine inflate | **FR-FCLC** | **0.9644 ± 0.0017** | **0.0946 ± 0.0002** | **0%** | **100%** |
| Byzantine deflate | Naive | 0.8972 ± 0.0061 | 0.2947 ± 0.0258 | 0% | 60% |
| Byzantine deflate | Robust | 0.7725 ± 0.0130 | 0.5368 ± 0.0211 | 0% | 0% |
| Byzantine deflate | Fair | 0.9615 ± 0.0003 | 0.0947 ± 0.0000 | 0% | 100% |
| Byzantine deflate | **FR-FCLC** | **0.9605 ± 0.0003** | **0.0947 ± 0.0000** | **0%** | **100%** |

**Statistical significance (FR-FCLC vs baselines, p < 0.05):**

| Comparison | Coverage ↑ | Gap ↓ |
|---|---|---|
| FR-FCLC vs Naive | ✓ all settings | ✓ all settings |
| FR-FCLC vs Robust | ✓ all settings | ✓ all settings |
| FR-FCLC vs Fair (coverage) | ✗ comparable | — |
| FR-FCLC vs Fair (gap) | — | ✓ honest + inflate |

Outputs saved to:
```
results/run_all_results.json
results/run_all_summary.json
results/byzantine_results.json
results/stats_analysis.json
plots/byzantine_robustness_inflate.png
plots/byzantine_robustness_deflate.png
```

---

## APS Scoring — Important Note

For BoolQ, APS scores are computed as a **binary restricted score**
normalized over the yes/no token probability mass only:

```
score = 1 − P(true_label) / (P(yes) + P(no))
```

This is necessary because Qwen2.5 spreads probability over 151,643 tokens.
Using standard full-vocabulary APS places "yes"/"no" tokens at rank ~9,000+,
causing all scores to saturate at 1.0.

**Confirmed Qwen2.5 token IDs for yes/no:**

| Token | ID |
|---|---|
| ' Yes' | 7414 |
| ' yes' | 9834 |
| 'yes' | 9693 |
| 'Yes' | 9454 |
| ' No' | 2308 |
| ' no' | 902 |
| 'no' | 2152 |
| 'No' | 2753 |

Run the diagnostic script to verify on your environment:
```bash
python debug_tokenizer.py
```

---

## Key Findings

1. **FR-FCLC is the only fair method at K=100** — coverage gap of 0.042 (below the 5% threshold). Naive gap is 0.263 (6× larger). Notably, the Fair baseline also fails the fairness threshold at K=100 (gap = 0.095) because its naive global floor is a poor fit for extreme non-IID distributions. FR-FCLC's robust floor provides a tighter anchor.

2. **Robust still fails at K=100 (78.1% coverage)** — the root cause is data heterogeneity, not scale. With Dirichlet(α=1.0) across 100 clients, some clients hold only 1–5 samples. Legitimate high scores from hard clients occupy the same upper tail as Byzantine inflate scores; trimming cannot distinguish them.

3. **FR-FCLC gap is stable under inflate attacks** — plateaus at ~0.095 through 50% corruption, demonstrating graceful degradation. Naive grows monotonically to 0.474.

4. **Deflate attacks are indistinguishable from easy clients** — FR-FCLC and Fair produce identical results under deflate (gap = 0.095). This is a fundamental limitation without explicit Byzantine detection.

5. **Statistically significant improvement over all baselines** — paired t-tests confirm FR-FCLC outperforms Naive and Robust on coverage and gap across all settings (p < 0.05). Against Fair, gap is significantly smaller in honest and inflate settings.

---

## Known Limitations

- **Robust fails due to data heterogeneity:** At K=100 with Dirichlet(α=1.0), extreme non-IID splits cause legitimate high scores to occupy the same tail as Byzantine inflate scores. Trimming cannot distinguish them, so Robust remains at 78.1% coverage.
- **Deflate indistinguishable from natural variation:** Per-client fairness adjustment cannot identify Byzantine low-scorers without an explicit detection mechanism.
- **Single federated round:** Multi-round convergence behavior has not been assessed.
- **TruthfulQA not validated at scale:** Full-vocabulary APS for TruthfulQA has not been tested with 100 clients.
