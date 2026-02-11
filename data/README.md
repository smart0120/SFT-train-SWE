# Data

Place your instruction-tuning dataset here or reference it in `config/train_config.yaml`.

## Format

Each example should have at least:
- `instruction`: user prompt
- `response`: assistant reply

Optional: use different column names and set `dataset.instruction_column` and `dataset.response_column` in config.

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

Set `dataset.swe_synth_config_path: "affinetes/environments/SWE-SYNTH/config.yaml"` and `dataset.format_response_agent_style: true` to use SWE-SYNTH system/instance templates and wrap the gold patch as THOUGHT + bash. Optionally use a different `dataset.system_prompt` for code-fixing (e.g. “You are a helpful assistant that fixes software bugs. Respond with a unified diff patch.”).
