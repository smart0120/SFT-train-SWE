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
  prepare_swe_for_qwen.py   # SWE tasks + code context → HF dataset (text column)
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

- **JSON (instruction/response):** e.g. `data/swe_synth_train.json`. Generate with `scripts/generate_swe_synth_dataset.py` (see `data/README.md`).
- **Pre-formatted text (with code context):** `scripts/prepare_swe_for_qwen.py` → HF dataset; set `dataset.use_text_column: true` and `paths.dataset_path` to that dataset dir.

Details and formats: `data/README.md`.

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
