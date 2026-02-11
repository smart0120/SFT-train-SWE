# Data

Place your instruction-tuning dataset here or reference it in `config/train_config.yaml`.

## Format

**Option A — instruction/response:** Each example has `instruction` (user prompt) and `response` (assistant reply). Optional: different column names via `dataset.instruction_column` and `dataset.response_column`.

**Option B — multi-turn messages:** Each example has `messages`: a list of `{"role": "system"|"user"|"assistant", "content": "..."}`. The trainer applies the tokenizer chat template to each example; no need to set `use_text_column`. Use this for code-fixing style (issue + code + patch) from `scripts/generate_swe_dataset.py`.

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

Set `dataset.swe_synth_config_path: "affinetes/environments/SWE-SYNTH/config.yaml"` and `dataset.format_response_agent_style: true` to use SWE-SYNTH system/instance templates and wrap the gold patch as THOUGHT + bash. Optionally use a different `dataset.system_prompt` for code-fixing.

## Combined: multi-turn messages JSON (code-fixing)

One script produces a **single JSON file** with multi-turn messages (closer to real code-fixing: system + user with issue/code + assistant patch). The trainer loads this JSON and uses `format_instruction` (chat template) on each example’s `messages` — keep `use_text_column: false`.

Input for `--source file` must be JSON/JSONL of **full task dicts** (e.g. `source`, `bug`, `original`), not the simplified instruction/response JSON. Use R2 or local cache to get task files, or save task dicts from your pipeline.

```bash
# From a task file (code-context = clone repos, add file content to user message)
python scripts/generate_swe_dataset.py path/to/tasks.json data/swe_dataset_messages.json --mode code-context --source file

# Simple mode (no repo): issue + patch only
python scripts/generate_swe_dataset.py path/to/tasks.json data/swe_dataset_messages.json --mode simple --source file

# From R2 (code-context)
python scripts/generate_swe_dataset.py --source r2 --task-range 0-99 data/swe_dataset_messages.json --mode code-context
```

Output format: each example is `{"instance_id": "...", "task_id": "..."|int, "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.

In `config/train_config.yaml`: set `paths.dataset_path: "data/swe_dataset_messages.json"` and leave `dataset.use_text_column: false`. Uses `config/swe_synth_agent.yaml` (or `--no-swe-synth-format` for simple Issue/Patch). Code-context mode requires `gitpython`; repos are cached under `data/repo_cache`.
