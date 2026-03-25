"""
data/preprocess.py
Downloads BoolQ and TruthfulQA, formats them into a unified schema,
and creates project-level train / calibration / test splits.

Unified schema per sample:
    {
        "input"     : str   - prompt-ready input string
        "target"    : str   - expected answer string
        "source"    : str   - "boolq" or "truthfulqa"
        "group"     : str   - used later for fairness evaluation
        "raw_split" : str   - original dataset split name
    }
"""

import os
import json
import random
import numpy as np
from datasets import load_dataset

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Output directory ─────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "processed")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Project split ratios ─────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
CALIB_RATIO = 0.15
TEST_RATIO  = 0.15   # must sum to 1.0


# ── BoolQ ─────────────────────────────────────────────────────────────────────
def process_boolq():
    """
    BoolQ: yes/no reading comprehension.
    Input  : structured Question + Context prompt
    Target : "yes" or "no"
    """
    print("Loading BoolQ...")
    ds = load_dataset("google/boolq")

    unified = []
    for split in ["train", "validation"]:
        for row in ds[split]:
            unified.append({
                "input":     f"Question: {row['question']}\nContext: {row['passage']}",
                "target":    "yes" if row["answer"] else "no",
                "source":    "boolq",
                "group":     "boolq",
                "raw_split": split
            })

    print(f"  BoolQ total samples: {len(unified)}")
    return unified


# ── TruthfulQA ────────────────────────────────────────────────────────────────
def process_truthfulqa():
    """
    TruthfulQA (generation): open-ended factual questions.
    Input  : Question prompt
    Target : best_answer string
    """
    print("Loading TruthfulQA...")
    ds = load_dataset("truthful_qa", "generation")

    unified = []
    for split in ["validation"]:
        for row in ds[split]:
            unified.append({
                "input":     f"Question: {row['question']}",
                "target":    row["best_answer"],
                "source":    "truthfulqa",
                "group":     "truthfulqa",
                "raw_split": split
            })

    print(f"  TruthfulQA total samples: {len(unified)}")
    return unified


# ── Project splits (train / calibration / test) ───────────────────────────────
def make_project_splits(data: list, name: str):
    """
    Ignore original dataset split names.
    Create our own train / calibration / test splits.
    Calibration split is used for conformal prediction threshold computation.
    """
    shuffled = data.copy()
    random.shuffle(shuffled)

    n       = len(shuffled)
    n_train = int(n * TRAIN_RATIO)
    n_calib = int(n * CALIB_RATIO)

    train = shuffled[:n_train]
    calib = shuffled[n_train : n_train + n_calib]
    test  = shuffled[n_train + n_calib:]

    print(f"\n  {name} project splits:")
    print(f"    train       : {len(train)}")
    print(f"    calibration : {len(calib)}")
    print(f"    test        : {len(test)}")

    return {"train": train, "calibration": calib, "test": test}


# ── Save to disk ──────────────────────────────────────────────────────────────
def save(data, name: str):
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


# ── Summary stats ─────────────────────────────────────────────────────────────
def print_summary(splits: dict, name: str):
    print(f"\n{name} summary:")
    for split_name, data in splits.items():
        if not data:
            continue
        targets = [d["target"] for d in data]
        unique  = set(targets)
        print(f"  {split_name:12s}: {len(data)} samples | "
              f"unique targets: {len(unique)} | "
              f"example target: '{targets[0]}'")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load and unify
    boolq      = process_boolq()
    truthfulqa = process_truthfulqa()

    # Create project-level splits
    boolq_splits      = make_project_splits(boolq,      "BoolQ")
    truthfulqa_splits = make_project_splits(truthfulqa, "TruthfulQA")

    # Save each split separately
    for split_name, data in boolq_splits.items():
        save(data, f"boolq_{split_name}")

    for split_name, data in truthfulqa_splits.items():
        save(data, f"truthfulqa_{split_name}")

    # Print summaries
    print_summary(boolq_splits,      "BoolQ")
    print_summary(truthfulqa_splits, "TruthfulQA")

    print("\nPreprocessing complete. Files saved in data/processed/")