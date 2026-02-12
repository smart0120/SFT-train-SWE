# Data

Place your instruction-tuning dataset here or reference it in `config/train_config.yaml`.

## Format

Each example must have **messages**: a list of `{"role": "system"|"user"|"assistant", "content": "..."}`. The trainer builds text from `messages` with the tokenizer chat template only (no extra formatting).

## Options

1. **Hugging Face dataset**: Set `dataset.name` in config (e.g. `"your_org/your_dataset"`).
2. **Local JSON**: Put a `.json` file in `data/` and set in config:
   - `paths.dataset_path: "data/train.json"`  
   or
   - `dataset.name: "json"` and `dataset.data_files: "data/train.json"`

See `data/sample_train.json` for a minimal example.

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
