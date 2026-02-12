"""
Run inference with a LoRA adapter or a merged model.
- Use --model_path pointing to outputs/merged/<run_name> for merged model.
- Use --base_model_id + --adapter_path for adapter-only (slower, no merge).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model_for_inference(
    model_path: str | Path | None = None,
    base_model_id: str | None = None,
    adapter_path: str | Path | None = None,
):
    """
    Load either:
    - Merged model: model_path = outputs/merged/<run_name>
    - Adapter only: base_model_id + adapter_path = outputs/adapters/<run_name>
    """
    if model_path:
        # Merged model (single directory)
        model_path = Path(model_path)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), token=os.environ.get("HF_TOKEN"))
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        return model, tokenizer

    if base_model_id and adapter_path:
        # Base + LoRA adapter
        adapter_path = Path(adapter_path)
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_path), token=os.environ.get("HF_TOKEN"))
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        model = model.merge_and_unload()
        return model, tokenizer

    raise ValueError("Provide either model_path (merged) or base_model_id + adapter_path")


def _inference_device(model):
    """Device for input tensors when using device_map='auto' (model may span devices)."""
    if hasattr(model, "device") and model.device is not None:
        return model.device
    return next(model.parameters()).device


def _to_input_tensor(inputs, tokenizer, device):
    """Normalize apply_chat_template result to a tensor on device (some tokenizers return str)."""
    if isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    if isinstance(inputs, str):
        out = tokenizer(inputs, return_tensors="pt")
        return out["input_ids"].to(device)
    return torch.tensor(inputs, dtype=torch.long).to(device)


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    do_sample: bool = True,
    temperature: float = 0.7,
):
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": prompt}]
    device = _inference_device(model)
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = _to_input_tensor(inputs, tokenizer, device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(1e-5, temperature),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def generate_chat(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 0.7,
) -> str:
    """
    Generate assistant reply given a list of messages (e.g. [{"role":"system","content":"..."}, {"role":"user","content":"..."}, ...]).
    Returns only the new assistant reply text (no prompt included).
    """
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    device = _inference_device(model)
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = _to_input_tensor(inputs, tokenizer, device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=max(1e-5, temperature),
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # Decode only the new tokens (assistant reply)
    input_length = inputs.shape[1] if inputs.dim() > 1 else inputs.shape[0]
    new_tokens = outputs[0][input_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default=None, help="Path to merged model (e.g. outputs/merged/llama-3.2-1b-lora)")
    p.add_argument("--base_model_id", type=str, default=None, help="Base model ID when using adapter_path")
    p.add_argument("--adapter_path", type=str, default=None, help="Path to adapter (e.g. outputs/adapters/llama-3.2-1b-lora)")
    p.add_argument("--prompt", type=str, default="What is the capital of France?")
    p.add_argument("--max_new_tokens", type=int, default=150)
    p.add_argument("--temperature", type=float, default=0.7)
    args = p.parse_args()

    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
    )
    out = generate(model, tokenizer, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(out)


if __name__ == "__main__":
    main()
