# FR-FCLC: Fair and Robust Federated Conformal Logit Calibration

> Mitigating Hallucinations in Federated Large Language Models via Fair and Robust Conformal Prediction

**Course:** Distributed AI — Spring 2026  
**Author:** Mirenge Innocent  
**Status:** Phase 4 of 6 complete — Robust aggregation and fairness-aware calibration implemented and verified  

---

## Overview

FR-FCLC is a post-hoc calibration framework for federated LLMs. It computes
Adaptive Prediction Set (APS) scores locally on each federated client and
aggregates them at the server to produce a global conformal threshold with
statistical coverage guarantees. Later phases extend this with Byzantine-robust
trimmed quantile aggregation and fairness-aware per-group threshold adjustment.

---

## Hardware Requirements

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
│   ├── fl_simulation.py       # Flower run_simulation entry point
│   └── __init__.py
├── conformal/                 # Phase 4: robust + fair aggregation (complete)
│   ├── aps_scores.py
│   ├── server_aggregate_naive.py
│   ├── robust_aggregate.py    # Trimmed quantile + Byzantine simulation
│   ├── fair_threshold.py      # Per-client fairness-aware threshold adjustment
│   ├── fr_fclc_pipeline.py    # Full integrated pipeline: all 4 methods × 3 settings
│   └── __init__.py
├── experiments/               # Phase 5: full evaluation scripts (upcoming)
│   ├── byzantine_simulation.py
│   ├── run_all.py
│   ├── stats_analysis.py
│   └── __init__.py
├── plots/                     # Generated figures
│   ├── boolq_client_distribution.png
│   ├── coverage_comparison.png    # Per-client coverage: all 4 methods (honest)
│   └── coverage_gap.png           # Coverage gap: all methods × all settings
├── results/                   # Simulation outputs (JSON)
│   ├── boolq_thresholds.json          # Naive global threshold per round
│   ├── boolq_client_scores.json       # Per-client APS scores from simulation
│   ├── fair_thresholds.json           # Per-client FR-FCLC thresholds
│   └── fr_fclc_results.json           # Full pipeline results: all methods × settings
├── debug_tokenizer.py         # Tokenizer diagnostic script
├── requirements.txt
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
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
| `N_CLIENTS` | 10 | Number of simulated clients |
| `ALPHA` | 1.0 | Dirichlet concentration (lower = more heterogeneous) |
| `MIN_SAMPLES` | 10 | Warn if any client has fewer samples |
| `MERGE_TINY` | False | Merge tiny clients into largest (use with α=0.5) |

> **Note on scaling:** For Phase 5 experiments (100 clients, α=0.5),
> set `MERGE_TINY=True` to prevent empty clients. Document this deviation
> from the raw Dirichlet split in experimental notes.

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
on 100 samples from client_000 (debug mode). Saves adapter to
`results/lora_client_000/`.

Expected output:
```
Trainable params : 3,686,400  (0.217% of total)
Avg loss    : ~2.07
Steps run   : 25
LoRA adapter saved → results/lora_client_000
```

---

### Phase 3: Federated Simulation and APS Calibration

**Step 5 — Run federated simulation**
```bash
python -m federated.fl_simulation
```

**Config options in `fl_simulation.py`:**

| Parameter | Current Value | Notes |
|---|---|---|
| `N_CLIENTS` | 2 | Scale up gradually: 2 → 5 → 10 → 100 |
| `N_ROUNDS` | 1 | Increase in Phase 5 |
| `DATASET` | "boolq" | Switch to "truthfulqa" when validated |

> **Memory warning:** Do not set `N_CLIENTS > 5` without first verifying
> memory usage. Each client loads a 6GB model. Ray is configured with
> `num_cpus=4` per client to limit parallel execution to 1–2 clients at
> a time.

Expected output after 1 round (2 clients):
```
[APS] n=191, mean=0.4476, std=0.2024, min=0.0000, max=1.000
[APS] n=191, mean=0.4618, std=0.1946, min=0.0062, max=1.000
[Server] Global APS scores — n=382, mean=0.4547, std=0.1987
[Server] Naive threshold (α=0.1) : 0.6905
```

Results saved to:
```
results/boolq_thresholds.json
results/boolq_client_scores.json
```

---

## Phase 4: Robust & Fair Aggregation (FR-FCLC Core)

### Step 6 — Byzantine-robust threshold (trimmed quantile)

```bash
python -m conformal.robust_aggregate
```

Loads `results/boolq_client_scores.json` and runs a full comparison of
naive vs trimmed quantile aggregation under honest and Byzantine attack settings.

**Config options in `robust_aggregate.py`:**

| Parameter | Value | Description |
|---|---|---|
| `ALPHA` | 0.1 | Miscoverage level (target coverage = 90%) |
| `TRIM_FRACTION` | 0.15 | Fraction trimmed from each tail |
| `BYZANTINE_FRACTION` | 0.30 | Fraction of clients simulated as Byzantine |

**Attack modes:** `"inflate"` (scores near 1.0), `"deflate"` (scores near 0.0), `"random"`

Expected output:
```
Honest naive threshold    : 0.6905
Honest robust threshold   : 0.5765
Byzantine naive threshold : 0.9908  (shift=0.3004)
Byzantine robust threshold: 0.9797  (shift=0.4031)
```

> **Note on scale:** Trimmed quantile robustness is meaningful only at sufficient
> client count. With 2 clients and 1 Byzantine (50% corruption), 15% trimming
> cannot protect the threshold. At 100 clients with 30% Byzantine (30 clients),
> trimming 15 from each tail eliminates all corrupted scores. Full robustness
> demonstration is planned for Phase 5.

---

### Step 7 — Fairness-aware per-client thresholds

```bash
python -m conformal.fair_threshold
```

Computes per-client thresholds using local calibration scores, with the
robust global threshold as a floor. Reports coverage gap as the fairness metric.

Expected output:
```
[Fair] Per-client taus    : client_1=0.7022, client_0=0.6725
[Fair] Fair thresholds    : client_1=0.7022, client_0=0.6725
[Fair] Coverage gap       : 0.0000 (FAIR)

Client         Naive Coverage  Fair Coverage
--------------------------------------------
client_000             0.7958         0.9058
client_001             0.7749         0.9058
--------------------------------------------
Coverage gap           0.0209         0.0000
Gap reduction: 0.0209
```

Output saved to `results/fair_thresholds.json`.

---

### Step 8 — Full FR-FCLC pipeline

```bash
python -m conformal.fr_fclc_pipeline
```

Runs all 4 methods (naive, robust, fair, FR-FCLC) across 3 settings
(honest, Byzantine inflate, Byzantine deflate) and generates plots.

**Methods compared:**

| Method | Robustness | Fairness |
|---|---|---|
| Naive | ✗ | ✗ |
| Robust | ✅ Trimmed quantile | ✗ |
| Fair | ✗ | ✅ Per-client threshold |
| FR-FCLC | ✅ Trimmed quantile | ✅ Per-client threshold |

**Phase 4 results summary (2 clients, 1 round):**

| Setting | Method | Cov Mean | Cov Gap | Fair | ≥90% |
|---|---|---|---|---|---|
| Honest | Naive | 0.9031 | 0.0052 | ✓ | ✓ |
| Honest | Robust | 0.7853 | 0.0209 | ✓ | ✗ |
| Honest | Fair | 0.9058 | 0.0000 | ✓ | ✓ |
| Honest | **FR-FCLC** | **0.9058** | **0.0000** | **✓** | **✓** |
| Byzantine (inflate) | Naive | 0.9031 | 0.1309 | ✗ | ✓ |
| Byzantine (inflate) | **FR-FCLC** | **0.9319** | **0.0524** | ✗ | ✓ |
| Byzantine (deflate) | Naive | 0.9031 | 0.1937 | ✗ | ✓ |
| Byzantine (deflate) | **FR-FCLC** | **0.9529** | **0.0942** | ✗ | ✓ |

Outputs saved to:
```
results/fr_fclc_results.json
plots/coverage_comparison.png
plots/coverage_gap.png
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

## Current Results Summary (Phase 4 Baseline — 2 clients, 1 round)

### Phase 3: APS Calibration Scores

| Metric | Client 000 | Client 001 | Global |
|---|---|---|---|
| Calibration samples | 191 | 191 | 382 |
| Mean APS score | 0.4476 | 0.4618 | 0.4547 |
| Std APS score | 0.2024 | 0.1946 | 0.1987 |
| Training samples | 454 | 2,036 | — |
| Training loss | 1.92 | 1.80 | — |
| Global threshold τ (α=0.1) | — | — | 0.6905 |

### Phase 4: FR-FCLC Pipeline (Honest Setting)

| Method | Coverage Mean | Coverage Gap | Fair | Meets ≥90% |
|---|---|---|---|---|
| Naive | 0.9031 | 0.0052 | ✓ | ✓ |
| Robust | 0.7853 | 0.0209 | ✓ | ✗ |
| Fair | 0.9058 | 0.0000 | ✓ | ✓ |
| **FR-FCLC** | **0.9058** | **0.0000** | **✓** | **✓** |

---

## Upcoming Phases

| Phase | Description | Status |
|---|---|---|
| Phase 5 | Scale to 10–100 clients, full evaluation across 5 seeds | Upcoming |
| Phase 6 | Final report, plots, reproducibility check, submission | Not started |

---

## Known Issues and Limitations

- **2-client limit:** Running more than 5 clients concurrently on M4 Pro
  risks memory pressure. Scale up gradually and monitor Activity Monitor.
- **Ray actor reloading:** Flower/Ray spawns new client actors for the
  evaluate phase, causing the model to reload (~20 sec per client). This
  is a known Flower virtual client limitation and not a code bug.
- **TruthfulQA APS not validated:** Full-vocabulary APS for TruthfulQA
  has not yet been tested at scale. Validation is planned for Phase 4.
- **Single federated round:** Convergence behavior across multiple rounds
  has not yet been assessed. Multi-round experiments are planned for Phase 5.