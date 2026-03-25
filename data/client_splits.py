"""
data/client_splits.py
Distributes training data across N federated clients using a Dirichlet
distribution (alpha) to simulate non-IID heterogeneity.

Strategy:
    BoolQ      → Dirichlet split by target class ("yes"/"no")
    TruthfulQA → Random split (targets are near-unique, Dirichlet not meaningful)

Scaling plan:
    Phase 2-3 debug   : N_CLIENTS=10,  ALPHA=1.0
    Phase 3 scale-up  : N_CLIENTS=20,  ALPHA=1.0
    Phase 4-5 full    : N_CLIENTS=100, ALPHA=1.0  (or ALPHA=0.5 with safeguards)

Safeguards:
    - Assertion fails if any client is empty (crash early, not silently)
    - Warning printed if any client has fewer than MIN_SAMPLES
    - Optional merge of tiny clients into largest (set MERGE_TINY=True)

Note on proposal alignment:
    The proposal targets Dirichlet α=0.5 at 100 clients.
    During implementation/debugging we use α=1.0 at smaller scale.
    When scaling to 100 clients with α=0.5, MERGE_TINY=True must be enabled
    and documented as: "clients below MIN_SAMPLES threshold were merged into
    the largest client to prevent empty-client failures in federated training."
"""

import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ── Config ─────────────────────────────────────────────────────────────────────
SEED        = 42
N_CLIENTS   = 10      # Phase 2-3 debug setting — increase later
ALPHA       = 1.0     # Use 0.5 only when scaling to 100 clients with MERGE_TINY=True
MIN_SAMPLES = 10      # Warn if any client has fewer samples than this
MERGE_TINY  = False   # Set True when using α=0.5 at 100 clients

DATA_DIR   = os.path.join(os.path.dirname(__file__), "processed")
CLIENT_DIR = os.path.join(DATA_DIR, "clients")

random.seed(SEED)
np.random.seed(SEED)


# ── Load / Save helpers ────────────────────────────────────────────────────────
def load_json(name: str) -> list:
    path = os.path.join(DATA_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ── Safeguards ─────────────────────────────────────────────────────────────────
def validate_buckets(client_buckets: list, min_samples: int):
    """
    Assert no empty clients exist.
    Warn about clients below min_samples threshold.
    """
    assert all(len(b) > 0 for b in client_buckets), (
        "Dirichlet split produced an empty client. "
        "Increase ALPHA, reduce N_CLIENTS, or set MERGE_TINY=True."
    )
    tiny = [i for i, b in enumerate(client_buckets) if len(b) < min_samples]
    if tiny:
        print(f"  Warning: {len(tiny)} clients have fewer than "
              f"{min_samples} samples: {tiny}")


# ── Merge tiny clients ─────────────────────────────────────────────────────────
def merge_small_clients(client_buckets: list, min_samples: int) -> list:
    """
    Merge clients with fewer than min_samples into the largest client.
    Only use when MERGE_TINY=True. Must be documented as a deviation
    from raw Dirichlet split in experimental notes.
    """
    large = [b for b in client_buckets if len(b) >= min_samples]
    small = [b for b in client_buckets if len(b) <  min_samples]

    if not small:
        return large

    largest_idx = int(np.argmax([len(b) for b in large]))
    for b in small:
        large[largest_idx].extend(b)

    print(f"  [MERGE] {len(small)} clients below threshold merged into "
          f"client {largest_idx:03d}. Active clients: {len(large)}")
    print(f"  [NOTE]  Document this in experimental notes as a safeguard "
          f"deviation from raw α={ALPHA} split.")
    return large


# ── Dirichlet split (BoolQ) ────────────────────────────────────────────────────
def dirichlet_split(data: list, n_clients: int, alpha: float) -> list:
    """
    Split data into n_clients subsets using Dirichlet over target classes.
    Only meaningful when targets are a small label set (e.g. yes/no).
    """
    class_indices = {}
    for idx, sample in enumerate(data):
        class_indices.setdefault(sample["target"], []).append(idx)

    for label in class_indices:
        random.shuffle(class_indices[label])

    client_buckets = [[] for _ in range(n_clients)]

    for label, indices in class_indices.items():
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, n_clients))
        counts = (proportions * len(indices)).astype(int)

        remainder = len(indices) - counts.sum()
        for i in np.random.choice(n_clients, remainder, replace=False):
            counts[i] += 1

        ptr = 0
        for cid, count in enumerate(counts):
            client_buckets[cid].extend([data[i] for i in indices[ptr: ptr + count]])
            ptr += count

    return client_buckets


# ── Random split (TruthfulQA) ──────────────────────────────────────────────────
def random_split(data: list, n_clients: int) -> list:
    """
    Randomly distribute data across n_clients.
    Used for TruthfulQA where targets are near-unique.
    """
    shuffled = data.copy()
    random.shuffle(shuffled)
    chunks = np.array_split(shuffled, n_clients)
    return [list(c) for c in chunks]


# ── Calibration split per client ───────────────────────────────────────────────
def assign_calibration(calib_data: list, n_clients: int) -> list:
    shuffled = calib_data.copy()
    random.shuffle(shuffled)
    chunks = np.array_split(shuffled, n_clients)
    return [list(c) for c in chunks]


# ── Debug: per-client class counts ────────────────────────────────────────────
def print_client_class_counts(client_buckets: list, n_show: int = 10):
    print(f"\n  Per-client class distribution (first {n_show} clients):")
    for cid, bucket in enumerate(client_buckets[:n_show]):
        counter = Counter(s["target"] for s in bucket)
        counts  = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
        print(f"    client {cid:03d}: total={len(bucket):4d} | {counts}")


# ── Visualize label distribution ───────────────────────────────────────────────
def plot_distribution(client_buckets: list, dataset_name: str,
                      n_clients: int, alpha: float):
    os.makedirs("plots", exist_ok=True)

    all_labels = sorted({s["target"] for b in client_buckets for s in b})
    if len(all_labels) > 5:
        counter    = Counter(s["target"] for b in client_buckets for s in b)
        all_labels = [l for l, _ in counter.most_common(5)]

    label_counts = {label: [] for label in all_labels}
    for bucket in client_buckets:
        bc = Counter(s["target"] for s in bucket)
        for label in all_labels:
            label_counts[label].append(bc.get(label, 0))

    fig, ax = plt.subplots(figsize=(14, 4))
    bottom  = np.zeros(n_clients)
    colors  = plt.cm.tab10.colors

    for i, label in enumerate(all_labels):
        counts = np.array(label_counts[label])
        ax.bar(range(n_clients), counts, bottom=bottom,
               label=label, color=colors[i % len(colors)], width=1.0)
        bottom += counts

    ax.set_xlabel("Client ID")
    ax.set_ylabel("Number of samples")
    ax.set_title(
        f"{dataset_name} — Non-IID label distribution "
        f"across {n_clients} clients (α={alpha})"
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out = f"plots/{dataset_name.lower()}_client_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Distribution plot saved → {out}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── BoolQ: Dirichlet split ─────────────────────────────────────────────────
    print("=" * 60)
    print(f"BoolQ — Dirichlet split (α={ALPHA}, {N_CLIENTS} clients)")
    print("=" * 60)

    boolq_train = load_json("boolq_train")
    boolq_calib = load_json("boolq_calibration")

    boolq_client_train = dirichlet_split(boolq_train, N_CLIENTS, ALPHA)

    if MERGE_TINY:
        boolq_client_train = merge_small_clients(boolq_client_train, MIN_SAMPLES)
    else:
        validate_buckets(boolq_client_train, MIN_SAMPLES)

    actual_n           = len(boolq_client_train)
    boolq_client_calib = assign_calibration(boolq_calib, actual_n)

    for cid in range(actual_n):
        base = os.path.join(CLIENT_DIR, "boolq", f"client_{cid:03d}")
        save_json(boolq_client_train[cid], os.path.join(base, "train.json"))
        save_json(boolq_client_calib[cid], os.path.join(base, "calibration.json"))

    sizes = [len(b) for b in boolq_client_train]
    print(f"\n  Active clients     : {actual_n}")
    print(f"  Train samples — min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}")

    print_client_class_counts(boolq_client_train, n_show=10)
    plot_distribution(boolq_client_train, "BoolQ", actual_n, ALPHA)

    # ── TruthfulQA: Random split ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TruthfulQA — Random split ({N_CLIENTS} clients)")
    print("=" * 60)

    tqa_train = load_json("truthfulqa_train")
    tqa_calib = load_json("truthfulqa_calibration")

    tqa_client_train = random_split(tqa_train, N_CLIENTS)
    tqa_client_calib = assign_calibration(tqa_calib, N_CLIENTS)

    for cid in range(N_CLIENTS):
        base = os.path.join(CLIENT_DIR, "truthfulqa", f"client_{cid:03d}")
        save_json(tqa_client_train[cid], os.path.join(base, "train.json"))
        save_json(tqa_client_calib[cid], os.path.join(base, "calibration.json"))

    sizes = [len(b) for b in tqa_client_train]
    print(f"\n  Active clients     : {N_CLIENTS}")
    print(f"  Train samples — min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.1f}")

    print("\nClient splits complete.")
    print(f"Data saved in: {CLIENT_DIR}")