"""
models/lora_finetune.py
Configures LoRA adapters on the Qwen2.5-3B-Instruct model
and runs a single fine-tuning loop on one client's data.

LoRA config:
    rank r=8, alpha=16, dropout=0.05
    target modules: q_proj, k_proj, v_proj, o_proj

Usage:
    from models.lora_finetune import build_peft_model, train_one_client
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from models.model_load import load_model_and_tokenizer, get_device

# ── Config ────────────────────────────────────────────────────────────────────
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_LENGTH = 256
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 2
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
DEBUG_SAMPLES = 100

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "clients"
)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ClientDataset(Dataset):
    """
    Wrap a client's JSON data into a PyTorch Dataset.
    Each sample becomes:
        "<input>\nAnswer: <target>"
    """

    def __init__(self, data: list, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = [f"{item['input']}\nAnswer: {item['target']}" for item in data]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.samples[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── LoRA setup ────────────────────────────────────────────────────────────────
def build_peft_model(model, device: str | None = None):
    """
    Wrap a model with LoRA adapters.

    On MPS, skip prepare_model_for_kbit_training because bitsandbytes training
    can be unstable on Apple Silicon.
    """
    if device is None:
        device = get_device()

    if device == "mps":
        print("  MPS detected — skipping k-bit preparation for stability")
    else:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.to(device)

    trainable_params = 0
    total_params = 0
    for _, parameter in peft_model.named_parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()

    print(
        f"  Trainable params : {trainable_params:,} "
        f"({100 * trainable_params / total_params:.3f}% of total)"
    )

    return peft_model


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_client(
    model,
    tokenizer,
    client_data: list,
    device: str | None = None,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM_STEPS,
    lr: float = LEARNING_RATE,
) -> dict:
    """
    Fine-tune the model on one client's data.
    Uses gradient accumulation for a larger effective batch size.
    """
    if device is None:
        device = get_device()

    print(f"  Training on device : {device}")
    print(
        f"  Batch size         : {batch_size} "
        f"(x{grad_accum} accum = {batch_size * grad_accum} effective)"
    )
    print(f"  Samples            : {len(client_data)}")

    dataset = ClientDataset(client_data, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=lr,
    )

    total_steps = max(1, ((len(dataloader) + grad_accum - 1) // grad_accum) * num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(10, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )

    model.train()
    total_loss = 0.0
    total_steps_run = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda parameter: parameter.requires_grad, model.parameters()),
                    1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_steps_run += 1

            if step % 10 == 0:
                print(
                    f"    Epoch {epoch + 1} | Step {step:3d}/{len(dataloader)} "
                    f"| Loss: {outputs.loss.item():.4f}"
                )

        # Handle leftover accumulated gradients
        if len(dataloader) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                1.0,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_steps_run += 1

        average_epoch_loss = epoch_loss / max(1, len(dataloader))
        total_loss += average_epoch_loss
        print(f"  Epoch {epoch + 1} complete | Avg loss: {average_epoch_loss:.4f}")

    return {
        "avg_loss": total_loss / max(1, num_epochs),
        "num_steps": total_steps_run,
        "num_samples": len(dataset),
    }


# ── Load client data ──────────────────────────────────────────────────────────
def load_client_data(dataset: str, client_id: int, split: str = "train") -> list:
    """Load one client's split from disk."""
    path = os.path.join(
        DATA_DIR,
        dataset,
        f"client_{client_id:03d}",
        f"{split}.json",
    )
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ── Main: single client test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LoRA Fine-tuning — Single Client Test (BoolQ, client_000)")
    print("=" * 60)

    model, tokenizer, device = load_model_and_tokenizer()

    print("\nApplying LoRA adapters...")
    model = build_peft_model(model, device)

    client_data = load_client_data("boolq", client_id=0, split="train")
    random.shuffle(client_data)

    if DEBUG_SAMPLES is not None:
        client_data = client_data[:DEBUG_SAMPLES]
        print(f"\nDebug mode: using {len(client_data)} samples from client_000")

    print("\nStarting training...")
    metrics = train_one_client(model, tokenizer, client_data, device=device)

    save_dir = "results/lora_client_000"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print(f"\nLoRA adapter saved → {save_dir}")

    print("\n" + "=" * 60)
    print("Training complete:")
    print(f"  Avg loss    : {metrics['avg_loss']:.4f}")
    print(f"  Steps run   : {metrics['num_steps']}")
    print(f"  Samples     : {metrics['num_samples']}")
    print("=" * 60)
    print("\nlora_finetune.py — OK")