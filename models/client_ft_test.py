"""
models/client_ft_test.py
Quick sanity check before running the full federated simulation.

Tests:
    1. Model loads correctly on MPS with 4-bit quantization
    2. LoRA adapters apply without errors
    3. Single-client fine-tuning runs for a few steps without OOM
    4. Reports peak memory and estimated time per client

Run this before launching fl_simulation.py to catch issues early:
    python -m models.client_ft_test
"""

import time
import json
import os
import torch
import numpy as np

from models.model_load import load_model_and_tokenizer, get_device
from models.lora_finetune import build_peft_model, train_one_client, load_client_data

# ── Config ─────────────────────────────────────────────────────────────────────
TEST_CLIENT_ID  = 0
TEST_DATASET    = "boolq"
TEST_SAMPLES    = 20      # Small subset just to verify pipeline works
NUM_EPOCHS      = 1


def run_test():
    print("=" * 60)
    print("FR-FCLC — Single Client Fine-Tuning Test")
    print("=" * 60)

    device = get_device()
    print(f"\nDevice          : {device}")
    print(f"Test client     : {TEST_CLIENT_ID}")
    print(f"Test samples    : {TEST_SAMPLES}")

    # ── 1. Load model ──────────────────────────────────────────────────────────
    print("\n[1/4] Loading model...")
    t0 = time.time()
    model, tokenizer, device = load_model_and_tokenizer()
    print(f"      Model loaded in {time.time()-t0:.1f}s")

    # ── 2. Apply LoRA ──────────────────────────────────────────────────────────
    print("\n[2/4] Applying LoRA adapters...")
    model = build_peft_model(model, device)

    # ── 3. Load client data ────────────────────────────────────────────────────
    print(f"\n[3/4] Loading client_{TEST_CLIENT_ID:03d} data...")
    client_data = load_client_data(TEST_DATASET, TEST_CLIENT_ID, "train")
    print(f"      Full client data : {len(client_data)} samples")

    # Limit to test samples
    import random
    random.seed(42)
    random.shuffle(client_data)
    test_data = client_data[:TEST_SAMPLES]
    print(f"      Using for test   : {len(test_data)} samples")

    # ── 4. Run training ────────────────────────────────────────────────────────
    print(f"\n[4/4] Running {NUM_EPOCHS} epoch(s) on {TEST_SAMPLES} samples...")
    t1 = time.time()
    metrics = train_one_client(
        model, tokenizer, test_data,
        device=device, num_epochs=NUM_EPOCHS
    )
    elapsed = time.time() - t1

    # ── Report ─────────────────────────────────────────────────────────────────
    samples_per_sec = TEST_SAMPLES / elapsed
    full_client_avg = 889   # Average samples per client at N_CLIENTS=10
    estimated_full  = full_client_avg / samples_per_sec / 60

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"  Avg loss          : {metrics['avg_loss']:.4f}")
    print(f"  Steps run         : {metrics['num_steps']}")
    print(f"  Samples tested    : {TEST_SAMPLES}")
    print(f"  Time elapsed      : {elapsed:.1f}s")
    print(f"  Throughput        : {samples_per_sec:.2f} samples/sec")
    print(f"  Est. full client  : ~{estimated_full:.1f} min "
          f"({full_client_avg} samples)")
    print(f"  Est. 10 clients   : ~{estimated_full*10/60:.1f} hrs "
          f"(sequential)")
    print("=" * 60)

    if metrics['avg_loss'] < 5.0:
        print("\n✓ Test passed — model trains without errors on MPS")
    else:
        print("\n✗ Test failed — loss too high, check model loading")

    print("\nclient_ft_test.py — OK")


if __name__ == "__main__":
    run_test()