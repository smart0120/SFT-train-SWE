"""
Merge LoRA adapter into the base model and save to outputs/merged/<run_name>.
Use this for deployment or faster inference without loading adapter separately.
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


def main(
    adapter_dir: str | Path,
    merged_output_dir: str | Path | None = None,
    model_id: str | None = None,
):
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    # Default merged dir: outputs/merged/<adapter_folder_name>
    if merged_output_dir is None:
        merged_output_dir = ROOT / "outputs" / "merged" / adapter_dir.name
    else:
        merged_output_dir = Path(merged_output_dir)

    # Get base model id from adapter config if not provided
    if model_id is None:
        import json
        adapter_config = adapter_dir / "adapter_config.json"
        if adapter_config.exists():
            with open(adapter_config) as f:
                cfg = json.load(f)
            model_id = cfg.get("base_model_name_or_path")
        if not model_id:
            raise ValueError("model_id required (not found in adapter_config.json)")

    merged_output_dir = merged_output_dir.resolve()
    merged_output_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, token=os.environ.get("HF_TOKEN"))
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(merged_output_dir)
    tokenizer.save_pretrained(merged_output_dir)

    print(f"Merged model saved to: {merged_output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model and save to outputs/merged/")
    p.add_argument("--adapter_dir", type=str, required=True, help="Path to saved adapter (e.g. outputs/adapters/llama-3.2-1b-lora)")
    p.add_argument("--merged_output_dir", type=str, default=None, help="Where to save merged model (default: outputs/merged/<adapter_dir_name>)")
    p.add_argument("--model_id", type=str, default=None, help="Base model ID (default: read from adapter_config.json)")
    args = p.parse_args()
    main(adapter_dir=args.adapter_dir, merged_output_dir=args.merged_output_dir, model_id=args.model_id)
