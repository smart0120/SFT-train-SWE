# LoRA SFT Training Project

End-to-end LoRA/QLoRA supervised fine-tuning for causal LMs (e.g. Llama 3.2 1B). Clear separation of **adapter storage** and **merged model storage**, with optional Weights & Biases logging.

## Directory layout

```
sft_lora/
├── config/
│   └── train_config.yaml    # Model, dataset, LoRA, training, paths
├── data/
│   ├── README.md
│   └── sample_train.json    # Example instruction/response pairs
├── outputs/
│   ├── adapters/            # LoRA adapter checkpoints and final adapter
│   │   └── <run_name>/      # e.g. llama-3.2-1b-lora
│   └── merged/              # Merged model (base + adapter) for deployment
│       └── <run_name>/
├── scripts/
│   ├── train.py                  # Train and save adapter to outputs/adapters/
│   ├── merge_adapter.py          # Merge adapter → outputs/merged/
│   ├── inference.py              # Run inference (adapter or merged)
│   └── generate_swe_synth_dataset.py  # Export SWE-SYNTH tasks → instruction/response JSON
├── src/
│   ├── config_loader.py
│   └── data_utils.py
├── .env.example             # Copy to .env and set secrets
├── requirements.txt
└── README.md
```

## Setup

### 1. Environment

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2. Environment variables and config

- Copy `.env.example` to `.env` and fill in:

| Variable | Purpose |
|----------|--------|
| `HF_TOKEN` | Hugging Face token (required for gated models like Llama) |
| `WANDB_API_KEY` | Weights & Biases API key (optional; set to `disabled` to turn off) |
| `WANDB_PROJECT` | W&B project name (e.g. `sft-lora`) |
| `WANDB_ENTITY` | W&B username or team |

- Edit `config/train_config.yaml`:
  - `model.model_id`: your base model (e.g. `meta-llama/Llama-3.2-1B-Instruct`)
  - `dataset.name` or `paths.dataset_path`: your dataset (see Data below)
  - `paths.adapter_output_dir` / `paths.merged_output_dir`: where to store adapters and merged models (defaults: `outputs/adapters`, `outputs/merged`)
  - `run_name`: used as subfolder under adapters and merged

### 3. Hugging Face (gated models)

```bash
huggingface-cli login
```

Accept the model’s terms on the model page if required.

## Data

- **Hugging Face**: set `dataset.name` in config (e.g. `"your_org/your_dataset"`).
- **Local JSON**: set `paths.dataset_path` to e.g. `data/sample_train.json` (or `data/train.json`). Each row must have `instruction` and `response` (or set `dataset.instruction_column` / `dataset.response_column`).
- **SWE-SYNTH**: run `python scripts/generate_swe_synth_dataset.py --task-range 0-99 --output data/swe_synth_train.json` to build a dataset from affinetes SWE-SYNTH tasks (task = problem_statement, correct answer = gold_patch). Then set `paths.dataset_path: "data/swe_synth_train.json"`. See `data/README.md`.

See `data/README.md` and `data/sample_train.json` for format.

## Usage

### Train (save adapter only)

Adapters are written to `outputs/adapters/<run_name>/`:

```bash
python scripts/train.py
# Or with custom config:
python scripts/train.py --config config/train_config.yaml
```

Override paths or model via env:

```bash
set MODEL_ID=meta-llama/Llama-3.2-1B-Instruct
set ADAPTER_OUTPUT_DIR=outputs/adapters
set RUN_NAME=my-run
python scripts/train.py
```

### Merge adapter into base model

Merge the saved adapter and store the full model under `outputs/merged/<run_name>/`:

```bash
python scripts/merge_adapter.py --adapter_dir outputs/adapters/llama-3.2-1b-lora
# Optional: specify merged output and base model
python scripts/merge_adapter.py --adapter_dir outputs/adapters/llama-3.2-1b-lora --merged_output_dir outputs/merged/llama-3.2-1b-lora --model_id meta-llama/Llama-3.2-1B-Instruct
```

### Inference

**Using merged model** (recommended):

```bash
python scripts/inference.py --model_path outputs/merged/llama-3.2-1b-lora --prompt "What is the capital of France?"
```

**Using adapter only** (loads base model + adapter):

```bash
python scripts/inference.py --base_model_id meta-llama/Llama-3.2-1B-Instruct --adapter_path outputs/adapters/llama-3.2-1b-lora --prompt "What is the capital of France?"
```

## Config reference

| Section | Key | Description |
|---------|-----|-------------|
| **paths** | `adapter_output_dir` | Where to save LoRA adapters |
| | `merged_output_dir` | Where to save merged models |
| | `dataset_path` | Optional local dataset path |
| **model** | `model_id` | Hugging Face model ID |
| | `use_4bit_quantization` | Use QLoRA (recommended for &lt;24GB GPU) |
| **lora** | `r`, `lora_alpha`, `target_modules`, etc. | LoRA hyperparameters |
| **training** | `report_to` | `"wandb"` or `"none"` |
| | `run_name` | Subfolder name under adapters/merged |

## Troubleshooting

- **CUDA OOM**: Lower `per_device_train_batch_size` and/or increase `gradient_accumulation_steps` in config.
- **Wandb off**: Set `WANDB_API_KEY=disabled` or `report_to: "none"` in config.
- **Gated model**: Run `huggingface-cli login` and accept the model terms on the model’s HF page.
