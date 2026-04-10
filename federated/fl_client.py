"""
federated/fl_client.py
Flower federated client for FR-FCLC.

Each client:
    1. Loads the base model ONCE in __init__ and reuses it across all calls
    2. Receives global LoRA adapter weights from server
    3. Fine-tunes locally on its own non-IID training data (fit)
    4. Computes calibration scores on local calibration data (evaluate)
    5. Returns updated LoRA weights + scores to server

Key runtime fix:
    The model is loaded once per client actor lifecycle, not reloaded
    inside fit() and evaluate().

BoolQ scoring:
    For BoolQ (yes/no), use a restricted binary score:
        score = 1 - P(true_label | {yes, no})
    where probabilities are normalized over the yes/no token sets only.

TruthfulQA scoring:
    For TruthfulQA, keep the fallback next-token APS-style score over the
    full vocabulary.
"""

import os
import json
import torch
import numpy as np
import flwr as fl
from collections import OrderedDict
from typing import Dict, List, Tuple

from models.lora_finetune import build_peft_model, train_one_client

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "clients"
)
MAX_LENGTH = 256
DEBUG_APS = True


# ── Load client data ──────────────────────────────────────────────────────────
def load_client_data(dataset: str, client_id: int, split: str) -> list:
    """Load one client split from disk."""
    path = os.path.join(
        DATA_DIR,
        dataset,
        f"client_{client_id:03d}",
        f"{split}.json",
    )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ── Calibration score computation ─────────────────────────────────────────────
def compute_aps_scores(
    model,
    tokenizer,
    calib_data: list,
    device: str,
    dataset: str = "boolq",
) -> np.ndarray:
    """
    Compute calibration scores for local calibration data.

    BoolQ:
        Restricted binary score over yes/no token sets:
            score = 1 - P(true_label | {yes, no})
        Lower score = more confident/correct.
        Higher score = less confident/wrong.

    TruthfulQA:
        Fallback full-vocabulary APS-style cumulative score.
    """
    model.eval()
    scores = []

    # Token sets discovered from debugging with Qwen
    yes_token_ids = {7414, 9834, 9693, 9454}   # ' Yes', ' yes', 'yes', 'Yes'
    no_token_ids = {2308, 902, 2152, 2753}     # ' No', ' no', 'no', 'No'

    if DEBUG_APS and dataset == "boolq":
        print(f"\n  [APS] YES token IDs: {sorted(yes_token_ids)}")
        print(f"  [APS] NO token IDs : {sorted(no_token_ids)}")

    with torch.no_grad():
        for index, item in enumerate(calib_data):
            try:
                if dataset == "boolq":
                    prompt = (
                        f"{item['input']}\n"
                        f"Answer with yes or no.\n"
                        f"Answer:"
                    )
                else:
                    prompt = f"{item['input']}\nAnswer:"

                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=False,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}

                outputs = model(**encoded)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=-1)

                if DEBUG_APS and index == 0:
                    top_probs, top_ids = torch.topk(probs, 5)
                    print(f"\n  [APS Debug] Top-5 predictions for first calibration sample:")
                    print(f"  Prompt tail: '...{prompt[-60:]}'")
                    for token_id, probability in zip(top_ids, top_probs):
                        token_text = tokenizer.decode([token_id.item()])
                        print(f"    {repr(token_text):12s} : {probability.item():.6f}")
                    print(f"  True target: {repr(item['target'])}")

                if dataset == "boolq":
                    target = item["target"].strip().lower()

                    p_yes = sum(
                        float(probs[token_id].item())
                        for token_id in yes_token_ids
                        if token_id < probs.shape[0]
                    )
                    p_no = sum(
                        float(probs[token_id].item())
                        for token_id in no_token_ids
                        if token_id < probs.shape[0]
                    )

                    denominator = p_yes + p_no
                    if denominator <= 0:
                        scores.append(1.0)
                        continue

                    p_yes_norm = p_yes / denominator
                    p_no_norm = p_no / denominator

                    if DEBUG_APS and index == 0:
                        print(
                            f"  [APS Binary Debug] "
                            f"p_yes={p_yes:.6f}, p_no={p_no:.6f}, "
                            f"p_yes_norm={p_yes_norm:.6f}, p_no_norm={p_no_norm:.6f}"
                        )

                    if target == "yes":
                        score = 1.0 - p_yes_norm
                    elif target == "no":
                        score = 1.0 - p_no_norm
                    else:
                        score = 1.0

                    scores.append(float(score))

                else:
                    target_ids = tokenizer.encode(
                        item["target"],
                        add_special_tokens=False,
                    )
                    if not target_ids:
                        scores.append(1.0)
                        continue

                    target_token_id = target_ids[0]
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    sorted_probs_np = sorted_probs.cpu().numpy()
                    sorted_indices_np = sorted_indices.cpu().numpy()

                    rank_positions = np.where(sorted_indices_np == target_token_id)[0]
                    if len(rank_positions) == 0:
                        scores.append(1.0)
                        continue

                    rank = rank_positions[0]
                    score = float(sorted_probs_np[: rank + 1].sum())
                    scores.append(min(score, 1.0))

            except Exception as error:
                print(f"  [APS] Error on sample {index}: {error}")
                scores.append(1.0)

    score_array = np.array(scores, dtype=np.float32)
    print(
        f"  [APS] n={len(score_array)}, "
        f"mean={score_array.mean():.4f}, "
        f"std={score_array.std():.4f}, "
        f"min={score_array.min():.4f}, "
        f"max={score_array.max():.4f}"
    )
    return score_array


# ── Flower Client ─────────────────────────────────────────────────────────────
class FRFCLCClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for FR-FCLC.

    The model is loaded once in __init__ and reused across fit/evaluate
    calls inside the same client actor instance.
    """

    def __init__(
        self,
        client_id: int,
        dataset: str,
        model,
        tokenizer,
        device: str,
    ):
        self.client_id = client_id
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        self.model = build_peft_model(model, device)

        self.train_data = load_client_data(dataset, client_id, "train")
        self.calib_data = load_client_data(dataset, client_id, "calibration")

        print(
            f"  Client {client_id:03d} ready | "
            f"train={len(self.train_data)}, calib={len(self.calib_data)}"
        )

    def _get_lora_keys(self) -> List[str]:
        """Return sorted LoRA parameter keys."""
        return sorted(
            key for key in self.model.state_dict().keys() if "lora_" in key
        )

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """Return LoRA parameters only."""
        state = self.model.state_dict()
        return [state[key].detach().cpu().numpy() for key in self._get_lora_keys()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Load LoRA parameters from server, preserving dtype/device."""
        keys = self._get_lora_keys()
        if len(parameters) != len(keys):
            raise ValueError(
                f"Client {self.client_id}: parameter mismatch — "
                f"got {len(parameters)}, expected {len(keys)}"
            )

        current_state = self.model.state_dict()
        updated = OrderedDict()

        for key, value in zip(keys, parameters):
            updated[key] = torch.tensor(
                value,
                dtype=current_state[key].dtype,
                device=current_state[key].device,
            )

        self.model.load_state_dict(updated, strict=False)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """Receive global LoRA weights, fine-tune locally, return updated weights."""
        self.set_parameters(parameters)

        metrics = train_one_client(
            self.model,
            self.tokenizer,
            self.train_data,
            device=self.device,
        )

        print(
            f"  Client {self.client_id:03d} fit | "
            f"loss={metrics['avg_loss']:.4f}, "
            f"samples={metrics['num_samples']}"
        )

        return self.get_parameters(config={}), metrics["num_samples"], {
            "loss": metrics["avg_loss"]
        }

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, int, Dict]:
        """Compute local calibration scores with the already-loaded model."""
        self.set_parameters(parameters)

        scores = compute_aps_scores(
            self.model,
            self.tokenizer,
            self.calib_data,
            self.device,
            dataset=self.dataset,
        )

        scores_str = ",".join(f"{score:.6f}" for score in scores)

        print(
            f"  Client {self.client_id:03d} eval | "
            f"calib={len(scores)}, mean={scores.mean():.4f}, std={scores.std():.4f}"
        )

        return float(scores.mean()), len(scores), {
            "aps_scores": scores_str,
            "client_id": self.client_id,
            "n_samples": len(scores),
        }