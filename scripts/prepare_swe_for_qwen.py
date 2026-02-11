"""
Prepare SWE-SYNTH / SWE-bench tasks for Qwen SFT with code context.

- Clones repos (cached per repo), reads file content at base_commit for modified files.
- Builds chat messages: system, user (issue + current code), assistant (patch in diff block).
- Applies the model's chat template to produce full prompt+completion text.
- Saves a Hugging Face Dataset with a "text" column for SFTTrainer(dataset_text_field="text").

Input: JSONL (one task per line), or JSON array, or --source r2/local with --task-range.
Task shape: SWE-SYNTH (source.repo, source.base_commit, bug.problem_statement, original.gold_patch)
            or SWE-bench (original.repo, original.base_commit, original.problem_statement, original.gold_patch)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

# Load .env for HF_TOKEN
_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from datasets import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

try:
    from git import Repo
except ImportError:
    Repo = None

try:
    from src.config_loader import load_config
except ImportError:
    load_config = None


def _get_default_model_id() -> str | None:
    """Model ID from training config (train_config.yaml) and MODEL_ID env override."""
    if os.environ.get("MODEL_ID"):
        return os.environ.get("MODEL_ID")
    if load_config is None:
        return None
    try:
        config = load_config(None)
        return (config.get("model") or {}).get("model_id")
    except Exception:
        return None

# Reuse R2/local loading when using task range
try:
    from generate_swe_synth_dataset import (
        DEFAULT_R2_PREFIX,
        DEFAULT_R2_URL,
        load_task_from_local,
        load_task_from_r2,
    )
except ImportError:
    DEFAULT_R2_URL = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev"
    DEFAULT_R2_PREFIX = "bugs"
    load_task_from_r2 = None
    load_task_from_local = None

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Patch parsing
# ----------------------------------------------------------------------
def get_modified_files(diff_text: str) -> list[str]:
    """Return list of file paths modified in a unified diff (from +++ b/ or --- a/)."""
    files = []
    for line in diff_text.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:].strip()
            if path and path != "/dev/null" and path not in files:
                files.append(path)
    return files


def load_file_at_commit(repo_path: str | Path, file_path: str, commit_hash: str) -> str | None:
    """Return content of a file at a specific commit, or None if not found."""
    if Repo is None:
        raise RuntimeError("Install gitpython: pip install gitpython")
    repo = Repo(repo_path)
    try:
        return repo.git.show(f"{commit_hash}:{file_path}")
    except Exception:
        return None


# ----------------------------------------------------------------------
# Task field extraction (SWE-SYNTH vs SWE-bench)
# ----------------------------------------------------------------------
def get_task_fields(instance: dict) -> dict | None:
    """
    Extract repo, base_commit, problem_statement, gold_patch from a task dict.
    Supports SWE-SYNTH (source/bug/original) and SWE-bench (original.*).
    """
    source = instance.get("source", {})
    bug = instance.get("bug", {})
    original = instance.get("original", {})

    repo = source.get("repo") or original.get("repo")
    base_commit = source.get("base_commit") or original.get("base_commit")
    problem_statement = (
        bug.get("problem_statement") or original.get("problem_statement") or ""
    ).strip()
    gold_patch = (original.get("gold_patch") or "").strip()

    if not repo or not base_commit or not problem_statement or not gold_patch:
        return None
    return {
        "repo": repo,
        "base_commit": base_commit,
        "problem_statement": problem_statement,
        "gold_patch": gold_patch,
        "instance_id": source.get("swe_instance_id") or instance.get("instance_id") or "unknown",
    }


def _normalize_problem_statement(text: str) -> str:
    s = (text or "").strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def _format_response_agent_style(patch: str) -> str:
    """Wrap gold patch as THOUGHT + single bash block (SWE-SYNTH agent style)."""
    if not patch or not patch.strip():
        return patch
    return (
        "THOUGHT: I will fix the issue by applying the following patch.\n\n"
        "```bash\n"
        "cat << 'PATCH' | git apply -\n"
        + patch.strip()
        + "\nPATCH\n"
        "```"
    )


def load_swe_synth_templates(config_path: str | Path) -> dict[str, str] | None:
    """
    Load agent.system_template and agent.instance_template from SWE-SYNTH config.yaml.
    Returns {"system_template": ..., "instance_template": ...} or None if missing/failed.
    """
    if yaml is None:
        LOG.warning("PyYAML not installed; cannot load SWE-SYNTH config")
        return None
    path = Path(config_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        LOG.warning("SWE-SYNTH config not found: %s", path)
        return None
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    agent = (config or {}).get("agent") or {}
    system_template = (agent.get("system_template") or "").strip()
    instance_template = (agent.get("instance_template") or "").strip()
    if not system_template or not instance_template:
        LOG.warning("SWE-SYNTH config missing agent.system_template or agent.instance_template")
        return None
    return {"system_template": system_template, "instance_template": instance_template}


# ----------------------------------------------------------------------
# Qwen-style formatting
# ----------------------------------------------------------------------
def format_for_qwen(
    problem_statement: str,
    code_context: str,
    gold_patch: str,
    tokenizer,
    system_prompt: str | None = None,
    swe_synth_templates: dict[str, str] | None = None,
) -> str:
    """
    Build chat messages and apply the tokenizer's chat template to get full prompt+completion text.

    If swe_synth_templates is set (from SWE-SYNTH config agent section), uses:
    - system: agent.system_template (THOUGHT + one bash block format)
    - user: agent.instance_template with {{task}} = problem_statement + current code
    - assistant: THOUGHT + bash block applying the patch (agent style)
    Otherwise uses the simple Issue/Current code/Patch format.
    """
    if swe_synth_templates:
        system_content = swe_synth_templates["system_template"]
        # {{task}} in instance_template gets the PR description + current code
        task_content = (
            f"{problem_statement}\n\n"
            "<current_code>\n"
            "Relevant file content at base_commit (pre-patch):\n"
            f"{code_context}\n"
            "</current_code>"
        )
        user_content = swe_synth_templates["instance_template"].replace("{{task}}", task_content)
        assistant_content = _format_response_agent_style(gold_patch)
    else:
        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant that writes patches for software issues. "
                "Given the issue description and the current code, generate a unified diff that solves the issue."
            )
        system_content = system_prompt
        user_content = f"Issue:\n{problem_statement}\n\nCurrent code:\n{code_context}".strip()
        assistant_content = f"Patch:\n```diff\n{gold_patch}\n```"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return text


# ----------------------------------------------------------------------
# Conversion
# ----------------------------------------------------------------------
def convert_instance(
    instance: dict,
    repo_cache_dir: str | Path,
    tokenizer,
    system_prompt: str | None = None,
    swe_synth_templates: dict[str, str] | None = None,
    max_code_tokens_per_file: int | None = None,
    clone_url_template: str = "https://github.com/{repo}.git",
) -> dict | None:
    """
    Convert one task to a row with "text" (full prompt+completion) and metadata.
    """
    fields = get_task_fields(instance)
    if fields is None:
        return None

    repo_name = fields["repo"]
    base_commit = fields["base_commit"]
    problem_statement = _normalize_problem_statement(fields["problem_statement"])
    gold_patch = fields["gold_patch"]
    instance_id = fields["instance_id"]

    repo_dir = os.path.join(repo_cache_dir, repo_name.replace("/", "_"))
    if not os.path.exists(repo_dir):
        if Repo is None:
            raise RuntimeError("Install gitpython: pip install gitpython")
        url = clone_url_template.format(repo=repo_name)
        LOG.info("Cloning %s -> %s", url, repo_dir)
        Repo.clone_from(url, repo_dir)

    modified_files = get_modified_files(gold_patch)
    code_parts = []
    for fpath in modified_files:
        content = load_file_at_commit(repo_dir, fpath, base_commit)
        if content is None:
            continue
        if max_code_tokens_per_file and hasattr(tokenizer, "__len__"):
            tokens = tokenizer.encode(content, add_special_tokens=False)
            if len(tokens) > max_code_tokens_per_file:
                content = tokenizer.decode(tokens[:max_code_tokens_per_file])
        code_parts.append(f"### File: {fpath}\n{content}")
    code_context = "\n".join(code_parts) if code_parts else "(no file content retrieved)"

    full_text = format_for_qwen(
        problem_statement,
        code_context,
        gold_patch,
        tokenizer,
        system_prompt=system_prompt,
        swe_synth_templates=swe_synth_templates,
    )

    return {
        "text": full_text,
        "repo": repo_name,
        "instance_id": instance_id,
        "gold_patch": gold_patch,
    }


def load_tasks_from_file(input_path: str | Path) -> list[dict]:
    """Load tasks from JSONL (one JSON per line) or JSON array."""
    path = Path(input_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    # Try JSONL first (each line is a JSON object)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if all(ln.startswith("{") for ln in lines):
        return [json.loads(ln) for ln in lines]
    # Single JSON array
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SWE-SYNTH/SWE-bench tasks for Qwen SFT (repo clone + code context + chat template)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input JSONL or JSON file with task dicts. Omit when using --source r2/local.",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output directory for Hugging Face dataset (default: data/swe_qwen_dataset)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for tokenizer (default: from config/train_config.yaml model.model_id or MODEL_ID env)",
    )
    parser.add_argument(
        "--repo_cache",
        default=None,
        help="Git repo cache directory (default: <project_root>/data/repo_cache)",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="Override system prompt for the chat (ignored if --swe_synth_config is set)",
    )
    parser.add_argument(
        "--swe_synth_config",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to agent config YAML (default: config/swe_synth_agent.yaml in project)",
    )
    parser.add_argument(
        "--no_swe_synth_format",
        action="store_true",
        help="Use simple Issue/Current code/Patch format instead of SWE-SYNTH agent templates",
    )
    parser.add_argument(
        "--max_code_tokens_per_file",
        type=int,
        default=None,
        help="Truncate each file's code to this many tokens (default: no limit)",
    )
    parser.add_argument(
        "--source",
        choices=("r2", "local", "file"),
        default="file",
        help="Task source: file (input path), r2, or local (default: file)",
    )
    parser.add_argument(
        "--task_range",
        type=str,
        default="0-99",
        metavar="START-END",
        help="Task ID range when source is r2 or local (default: 0-99)",
    )
    parser.add_argument(
        "--r2_url",
        type=str,
        default=DEFAULT_R2_URL,
        help="R2 base URL when source=r2",
    )
    parser.add_argument(
        "--r2_prefix",
        type=str,
        default=DEFAULT_R2_PREFIX,
        help="R2 prefix when source=r2",
    )
    parser.add_argument(
        "--local_cache",
        type=Path,
        default=Path("/tmp/swe-synth-cache"),
        help="Local task cache dir when source=local",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    model_id = args.model or _get_default_model_id()
    if not model_id:
        parser.error(
            "Model not set. Use --model <name> or set MODEL_ID in .env, or set model.model_id in config/train_config.yaml"
        )
    LOG.info("Using tokenizer for model: %s", model_id)

    out_dir = args.output or str(ROOT / "data" / "swe_qwen_dataset")
    repo_cache = args.repo_cache or str(ROOT / "data" / "repo_cache")
    os.makedirs(repo_cache, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=os.environ.get("HF_TOKEN"), trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    swe_synth_templates = None
    if not args.no_swe_synth_format:
        config_path = args.swe_synth_config or (ROOT / "config" / "swe_synth_agent.yaml")
        swe_synth_templates = load_swe_synth_templates(config_path)
        if swe_synth_templates:
            LOG.info("Using SWE-SYNTH format from %s", config_path)
        else:
            swe_synth_templates = None

    # Load task list
    if args.source == "file":
        if not args.input:
            parser.error("input file required when source=file")
        instances = load_tasks_from_file(args.input)
        LOG.info("Loaded %s tasks from %s", len(instances), args.input)
    else:
        start, end = 0, 99
        if args.task_range:
            parts = args.task_range.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0]), int(parts[1])
                except ValueError:
                    pass
        if start > end:
            start, end = end, start
        instances = []
        for task_id in range(start, end + 1):
            if args.source == "r2" and load_task_from_r2:
                task = load_task_from_r2(task_id, args.r2_url, args.r2_prefix)
            elif args.source == "local" and load_task_from_local:
                task = load_task_from_local(task_id, args.local_cache)
            else:
                task = None
            if task is not None:
                instances.append(task)
            if args.source == "r2" and task_id < end:
                time.sleep(0.3)
        LOG.info("Loaded %s tasks from %s (range %s-%s)", len(instances), args.source, start, end)

    converted = []
    for inst in tqdm(instances, desc="Converting"):
        try:
            row = convert_instance(
                inst,
                repo_cache,
                tokenizer,
                system_prompt=args.system_prompt,
                swe_synth_templates=swe_synth_templates,
                max_code_tokens_per_file=args.max_code_tokens_per_file,
            )
            if row is not None:
                converted.append(row)
        except Exception as e:
            fid = (inst.get("source") or {}).get("swe_instance_id") or (get_task_fields(inst) or {}).get("instance_id") or "?"
            LOG.warning("Skipping %s: %s", fid, e)

    dataset = Dataset.from_list(converted)
    dataset.save_to_disk(out_dir)
    LOG.info("Saved %s examples to %s", len(dataset), out_dir)


if __name__ == "__main__":
    main()
