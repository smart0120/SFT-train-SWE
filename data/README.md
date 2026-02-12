# Data

Place your instruction-tuning dataset here or reference it in `config/train_config.yaml`.

## Format

Each example must have **messages**: a list of `{"role": "system"|"user"|"assistant", "content": "..."}`. The trainer builds text from `messages` with the tokenizer chat template only (no extra formatting).

## Options

1. **Hugging Face dataset**: Set `dataset.dataset_hf_id` in config (e.g. `"Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"`). The dataset must have a `messages` column; no local file needed.
2. **Local JSON**: Put a `.json` file in `data/` and set `paths.dataset_path: "data/train.json"`.

See `data/sample_train.json` for a minimal example.

## mini-SWE-agent-plus (66k trajectories)

The [mini-SWE-agent-plus](https://mini-swe-agent.com/) trajectory dataset on HuggingFace is already in the right format (`instance_id` + `messages`). Use it with SFT + LoRA by setting in `config/train_config.yaml`:

```yaml
paths:
  dataset_path: null   # or omit; not used when dataset_hf_id is set

dataset:
  dataset_hf_id: "Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k"
  split: "train"
  assistant_only_loss: true
```

Then run `python scripts/train.py`. For 66k examples you may want fewer epochs (e.g. `num_train_epochs: 3`) or a subset; adjust `training.*` in config as needed.

## SWE-SYNTH (task / eval → instruction, correct answer → response)

Tasks from **affinetes/environments/SWE-SYNTH** use:
- **Task / eval**: `problem_statement` (user-facing bug report) → use as `instruction`
- **Correct answer**: `gold_patch` (the fix diff) → use as `response`

Generated `swe_synth_train.json` entries look like:

```json
{
  "instruction": "Markdown links with underscores break formatting...",
  "response": "diff --git a/src/Markdown.ts ...",
  "task_id": 1020
}
```

- `instruction`: normalized problem statement (surrounding quotes stripped)
- `response`: unified diff (gold patch)
- `task_id`: source task id for tracing (optional for training)

To generate the SFT dataset from SWE-SYNTH:

```bash
# From project root: fetch tasks from public R2 and save to data/swe_synth_train.json
python scripts/generate_swe_synth_dataset.py --task-range 0-99 --output data/swe_synth_train.json

# From local cache (e.g. after running SWE-SYNTH eval)
python scripts/generate_swe_synth_dataset.py --source local --local-cache /tmp/swe-synth-cache --output data/swe_synth_train.json
```

Then set in `config/train_config.yaml`:

```yaml
paths:
  dataset_path: "data/swe_synth_train.json"
```

## Multi-turn messages JSON (code-fixing)

One script produces a **single JSON file** with multi-turn messages. The trainer expects `messages` only; set `paths.dataset_path: "data/swe_dataset_messages.json"`.

Input for `--source file` must be JSON/JSONL of **full task dicts** (e.g. `source`, `bug`, `original`). Use R2 or local cache to get task files, or save task dicts from your pipeline.

```bash
# From a task file (code-context = clone repos, add file content to user message)
python scripts/generate_swe_dataset.py path/to/tasks.json data/swe_dataset_messages.json --mode code-context --source file

# Simple mode (no repo): issue + patch only
python scripts/generate_swe_dataset.py path/to/tasks.json data/swe_dataset_messages.json --mode simple --source file

# From R2 (code-context)
python scripts/generate_swe_dataset.py --source r2 --task-range 0-99 data/swe_dataset_messages.json --mode code-context
```

Output: each example is `{"instance_id": "...", "task_id": ... , "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}`. Code-context mode uses `config/swe_synth_agent.yaml` (or `--no-swe-synth-format` for simple); repos are cached under `data/repo_cache`.
