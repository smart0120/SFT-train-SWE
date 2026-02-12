# LoRA SFT Training

LoRA/QLoRA supervised fine-tuning for causal LMs. Adapters go to `outputs/adapters/<run_name>`, merged models to `outputs/merged/<run_name>`.

## Layout

```
config/train_config.yaml   # Model, dataset, LoRA, training
data/                      # Datasets (see data/README.md)
outputs/adapters/          # LoRA checkpoints
outputs/merged/            # Merged model (base + adapter)
scripts/
  train.py                 # Train → save adapter
  merge_adapter.py         # Merge adapter into base
  inference.py             # CLI inference
  webui.py                 # Chat UI (gradio)
  generate_swe_synth_dataset.py   # SWE-SYNTH → instruction/response JSON
  generate_swe_dataset.py         # Multi-turn messages (code-fixing), simple or code-context → single JSON
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows: source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, set `HF_TOKEN` (and optionally `MODEL_ID`, `WANDB_*`). For gated models: `huggingface-cli login`.

## Config

Edit `config/train_config.yaml`: `model.model_id`, `paths.dataset_path`, `run_name`. Env vars in `.env` override config (e.g. `MODEL_ID`).

## Data

Datasets are **messages-only**: each example has `messages` (list of `{"role": "system"|"user"|"assistant", "content": "..."}`). Set `paths.dataset_path` to a JSON file (or HF dataset dir). The trainer builds training text from each example’s `messages` using the tokenizer chat template.

To generate SWE code-fixing data: `python scripts/generate_swe_dataset.py [input] [output] --mode code-context --source file`. See `data/README.md`.

## Commands

```bash
python scripts/train.py
python scripts/merge_adapter.py --adapter_dir outputs/adapters/<run_name>
python scripts/inference.py --model_path outputs/merged/<run_name> --prompt "Your prompt"
python scripts/webui.py   # optional; pip install gradio
```

## Troubleshooting

- **OOM:** Lower `per_device_train_batch_size` or set `use_4bit_quantization: true`.
- **Wandb:** Set `WANDB_API_KEY=disabled` or `report_to: "none"`.
- **Gated model:** `huggingface-cli login` and accept terms on the model page.
