"""
LoRA/QLoRA SFT training script.
Uses config/train_config.yaml and .env for HF token, wandb, and paths.
Saves adapters to outputs/adapters/<run_name>; merge separately to outputs/merged/.
"""
import os
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present (for HF_TOKEN, WANDB_*)
_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

try:
    from trl import SFTConfig
except ImportError:
    SFTConfig = None

from src.config_loader import load_config
from src.data_utils import load_and_prepare_dataset, format_instruction


def _from_pretrained_causal_lm(model_id, **kwargs):
    """Load causal LM; use dtype (new) or torch_dtype (legacy) per transformers API."""
    torch_dtype = kwargs.pop("torch_dtype", None)
    if torch_dtype is not None:
        try:
            return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
        except TypeError:
            return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


class WandbEpochLossCallback(TrainerCallback):
    """Log average train loss per epoch to wandb so you can plot loss vs epoch (x-axis = epoch)."""

    def __init__(self):
        self._epoch_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self._epoch_losses.append(logs["loss"])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self._epoch_losses:
            try:
                import wandb
                avg_loss = sum(self._epoch_losses) / len(self._epoch_losses)
                wandb.log({"train/loss_epoch": avg_loss, "epoch": state.epoch})
            except Exception:
                pass
            self._epoch_losses = []


def main(config_path: str | None = None):
    config = load_config(config_path)

    paths = config["paths"]
    model_cfg = config["model"]
    dataset_cfg = config["dataset"]
    lora_cfg = config["lora"]
    training_cfg = config["training"]
    run_name = config.get("run_name", "lora-run")

    adapter_dir = (ROOT / paths["adapter_output_dir"]).resolve()
    run_adapter_dir = adapter_dir / run_name
    run_adapter_dir.mkdir(parents=True, exist_ok=True)
    training_cfg["output_dir"] = str(run_adapter_dir)

    model_id = model_cfg["model_id"]
    print(f"Base model: {model_id}")
    use_4bit = model_cfg.get("use_4bit_quantization", True)
    dtype_name = model_cfg.get("torch_dtype", "bfloat16")
    torch_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16

    # Tokenizer
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ.get("HF_TOKEN"))
    tokenizer.pad_token = tokenizer.eos_token
    truncation_side = training_cfg.get("truncation_side", "right")
    if hasattr(tokenizer, "truncation_side"):
        tokenizer.truncation_side = truncation_side

    # Model (with optional QLoRA)
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"Loading model (4-bit) for {model_id}...")
        model = _from_pretrained_causal_lm(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch_dtype,
            token=os.environ.get("HF_TOKEN"),
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model for {model_id}...")
        model = _from_pretrained_causal_lm(
            model_id,
            device_map="auto",
            torch_dtype=torch_dtype,
            token=os.environ.get("HF_TOKEN"),
        )

    # LoRA
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    dataset_path = paths.get("dataset_path") or dataset_cfg.get("dataset_path")
    if dataset_path and not Path(dataset_path).is_absolute():
        dataset_path = str(ROOT / dataset_path)
    dataset = load_and_prepare_dataset(
        dataset_name=dataset_cfg["name"],
        split=dataset_cfg.get("split", "train"),
        instruction_column=dataset_cfg.get("instruction_column", "instruction"),
        response_column=dataset_cfg.get("response_column", "response"),
        dataset_path=dataset_path,
        data_files=dataset_cfg.get("data_files"),
    )
    validation_ratio = float(training_cfg.get("validation_ratio", 0))
    if validation_ratio > 0 and validation_ratio < 1:
        split = dataset.train_test_split(test_size=validation_ratio, seed=training_cfg.get("seed", 42))
        dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Train/val split: {len(dataset)} train, {len(eval_dataset)} validation ({validation_ratio:.0%} val)")
    else:
        eval_dataset = None
        if validation_ratio != 0:
            print("validation_ratio must be in (0, 1); skipping validation.")
    system_prompt = dataset_cfg.get("system_prompt", "You are a helpful assistant.")
    instance_template = dataset_cfg.get("instance_template")
    format_response_agent_style = dataset_cfg.get("format_response_agent_style", False)

    def formatting_func(example):
        return format_instruction(
            example,
            tokenizer,
            system_prompt=system_prompt,
            instance_template=instance_template,
            format_response_agent_style=format_response_agent_style,
        )

    # Wandb: set env from config so report_to="wandb" uses correct project/entity
    wandb_project = training_cfg.get("wandb_project")
    wandb_entity = training_cfg.get("wandb_entity")
    if wandb_project:
        os.environ.setdefault("WANDB_PROJECT", str(wandb_project))
    if wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", str(wandb_entity))

    report_to = training_cfg.get("report_to", "none")
    if report_to == "wandb" and os.environ.get("WANDB_API_KEY") in (None, "", "disabled"):
        report_to = "none"

    _common_args = dict(
        output_dir=training_cfg["output_dir"],
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=float(training_cfg.get("learning_rate", 2e-4)),
        fp16=training_cfg.get("fp16", False),
        bf16=training_cfg.get("bf16", True),
        logging_steps=training_cfg.get("logging_steps", 10),
        save_strategy=training_cfg.get("save_strategy", "steps"),
        save_steps=training_cfg.get("save_steps", 100),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        optim=training_cfg.get("optim", "adamw_torch"),
        report_to=report_to,
        run_name=run_name,
    )
    if eval_dataset is not None:
        _common_args["eval_strategy"] = training_cfg.get("eval_strategy", "steps")
        _common_args["eval_steps"] = training_cfg.get("eval_steps", 100)
    else:
        _common_args["eval_strategy"] = "no"
    if SFTConfig is not None:
        sft_config = SFTConfig(
            **_common_args,
            max_length=training_cfg.get("max_seq_length", 1024),
            packing=training_cfg.get("packing", False),
        )
    else:
        sft_config = TrainingArguments(**_common_args)

    callbacks = []
    if report_to == "wandb":
        callbacks.append(WandbEpochLossCallback())

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        callbacks=callbacks,
    )

    trainer.train()
    trainer.save_model(str(run_adapter_dir))
    tokenizer.save_pretrained(str(run_adapter_dir))

    print(f"\nAdapter saved to: {run_adapter_dir}")
    print(f"To merge and save full model, run: python scripts/merge_adapter.py --adapter_dir {run_adapter_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None, help="Path to YAML config (default: config/train_config.yaml)")
    args = p.parse_args()
    main(config_path=args.config)
