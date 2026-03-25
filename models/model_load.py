"""
models/model_load.py
Loads Qwen2.5-3B-Instruct with the best available precision
for Apple M4 Pro (MPS backend).

Loading strategy:
    1. Try 4-bit quantization via bitsandbytes (most memory efficient)
    2. Fall back to float16 on MPS
    3. Fall back to float32 on CPU

Usage:
    from models.model_load import load_model_and_tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_LENGTH = 512


# ── Device detection ──────────────────────────────────────────────────────────
def get_device() -> str:
    """Return the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Tokenizer ─────────────────────────────────────────────────────────────────
def load_tokenizer(model_id: str = MODEL_ID):
    """Load tokenizer and ensure a valid pad token exists."""
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"  Vocab size     : {tokenizer.vocab_size}")
    print(f"  Pad token      : '{tokenizer.pad_token}'")
    print(f"  Max model len  : {tokenizer.model_max_length}")
    return tokenizer


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model(model_id: str = MODEL_ID, device: str | None = None):
    """Load the model with the best available precision for the device."""
    if device is None:
        device = get_device()

    print(f"\nLoading model : {model_id}")
    print(f"Target device : {device}")

    # Attempt 1: 4-bit quantization for CUDA/MPS
    if device in ("cuda", "mps"):
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                dtype=torch.float16,
            )
            print("  Loaded with 4-bit quantization (bitsandbytes NF4)")
            return model, "4bit"

        except Exception as error:
            print(f"  4-bit quantization failed: {error}")
            print("  Falling back to non-quantized loading...")

    # Attempt 2: float16 on MPS
    if device == "mps":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=torch.float16,
            )
            model.to(device)
            print("  Loaded with float16 on MPS")
            return model, "float16-mps"

        except Exception as error:
            print(f"  float16 MPS failed: {error}")
            print("  Falling back to float32 on CPU...")

    # Attempt 3: float32 on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32,
    )
    model.to("cpu")
    print("  Loaded with float32 on CPU (slow)")
    return model, "float32-cpu"


# ── Combined loader ───────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_id: str = MODEL_ID):
    """Load tokenizer, model, and resolved device."""
    device = get_device()
    tokenizer = load_tokenizer(model_id)
    model, mode = load_model(model_id, device)

    n_params = sum(parameter.numel() for parameter in model.parameters()) / 1e9
    print(f"\n  Parameters     : {n_params:.2f}B")
    print(f"  Loading mode   : {mode}")
    print(f"  Device         : {device}")

    return model, tokenizer, device


# ── Quick inference test ──────────────────────────────────────────────────────
def test_inference(model, tokenizer, device: str | None = None):
    """Run a quick deterministic inference test."""
    if device is None:
        device = get_device()

    print("\nRunning inference test...")
    prompt = (
        "Question: Is the sky blue?\n"
        "Context: The sky appears blue due to Rayleigh scattering.\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    print(f"  Prompt   : {prompt}")
    print(f"  Response : {response}")
    return response


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tokenizer, device = load_model_and_tokenizer()
    test_inference(model, tokenizer, device)
    print("\nmodel_load.py — OK")