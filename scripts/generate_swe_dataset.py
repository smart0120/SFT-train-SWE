"""
Combined SWE dataset generator: multi-turn messages (code-fixing style), JSON output.

- Modes: simple (issue + patch only) or code-context (issue + current code + patch, with repo clone).
- Task source: file (JSON/JSONL), r2, or local cache.
- Output: single JSON file. Each item: {"instance_id": str, "task_id": str|int, "messages": [{"role": "system"|"user"|"assistant", "content": "..."}, ...]}.
- Trainer uses this via dataset_path + format_instruction (messages) — no use_text_column needed.
"""

import argparse
import json
import logging
import os
import shutil
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

try:
    from git import Repo
except ImportError:
    Repo = None

LOG = logging.getLogger(__name__)

NO_CODE_PLACEHOLDER = "(no code context provided)"


# ---------------------------------------------------------------------------
# Helpers (task fields, patch parsing, repo, templates)
# ---------------------------------------------------------------------------
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


def _parse_code_context(code_context: str) -> list[tuple[str, str]]:
    """Parse code_context string (### File: path\\ncontent...) into [(path, content), ...]."""
    if not code_context or code_context.strip() == NO_CODE_PLACEHOLDER:
        return []
    out = []
    for block in code_context.split("### File:")[1:]:  # skip leading empty
        block = block.strip()
        if not block:
            continue
        first_newline = block.find("\n")
        if first_newline >= 0:
            path, content = block[:first_newline].strip(), block[first_newline + 1 :].strip()
        else:
            path, content = block, ""
        if path:
            out.append((path, content))
    return out


def _first_user_task_only(
    problem_statement: str,
    swe_synth_templates: dict[str, str] | None,
    system_prompt: str | None,
) -> str:
    """First user message: PR description + instructions only (no file structure/code dump)."""
    if swe_synth_templates:
        task_content = problem_statement
        return swe_synth_templates["instance_template"].replace("{{task}}", task_content)
    if system_prompt is None:
        system_prompt = (
            "You are an AI assistant that writes patches for software issues. "
            "Given the issue description and the current code, generate a unified diff that solves the issue."
        )
    return f"Issue:\n{problem_statement}".strip()


def build_messages_pingpong(
    problem_statement: str,
    code_context: str,
    gold_patch: str,
    system_prompt: str | None = None,
    swe_synth_templates: dict[str, str] | None = None,
) -> list[dict]:
    """
    Build ping-pong: each assistant turn = exactly one bash; next user = result of that bash (returncode + output).
    Flow: user (task only) -> asst (one bash) -> user (print result) -> asst (one bash) -> user (print result) -> ...
    """
    system_content = (
        swe_synth_templates["system_template"]
        if swe_synth_templates
        else (system_prompt or "You are an AI assistant that writes patches for software issues.")
    )
    first_user = _first_user_task_only(problem_statement, swe_synth_templates, system_prompt)
    files_list = _parse_code_context(code_context)
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": first_user},
    ]

    # Step 1: assistant runs one bash (explore), user replies with result
    messages.append({
        "role": "assistant",
        "content": (
            "THOUGHT: I need to see the repository layout first.\n\n"
            "```bash\n"
            "ls -la\n"
            "```"
        ),
    })
    ls_output = (
        "total 148\ndrwxr-sr-x 1 root 1001  4096 Aug  6  2025 .\ndrwxr-xr-x 1 root root  4096 Feb 11 17:06 ..\ndrwxr-sr-x 1 root 1001  4096 Aug  6  2025 .git\n-rw-r--r-- 1 root 1001 5213 Aug  6  2025 README.md\ndrwxr-sr-x 1 root 1001  4096 Aug  6  2025 lib\ndrwxr-sr-x 1 root 1001  4096 Aug  6  2025 test\n"
        if files_list
        else "(no directory listing)"
    )
    messages.append({
        "role": "user",
        "content": f"<returncode>0</returncode>\n<output>\n{ls_output}\n</output>",
    })

    # Step 2..N: for each relevant file, assistant runs cat <path>, user replies with file content
    for path, content in files_list:
        messages.append({
            "role": "assistant",
            "content": (
                "THOUGHT: Reading the file that may need to be modified.\n\n"
                "```bash\n"
                f"cat {path}\n"
                "```"
            ),
        })
        messages.append({
            "role": "user",
            "content": f"<returncode>0</returncode>\n<output>\n{content}\n</output>",
        })

    # Next: assistant applies patch (one bash), user replies with result
    messages.append({
        "role": "assistant",
        "content": _format_response_agent_style(gold_patch),
    })
    messages.append({
        "role": "user",
        "content": "<returncode>0</returncode>\n<output>\nPatch applied successfully.\n</output>",
    })

    # Final: assistant submits (one bash)
    messages.append({
        "role": "assistant",
        "content": (
            "THOUGHT: Submitting the fix.\n\n"
            "```bash\n"
            "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n"
            "```"
        ),
    })
    return messages


def build_messages(
    problem_statement: str,
    code_context: str,
    gold_patch: str,
    system_prompt: str | None = None,
    swe_synth_templates: dict[str, str] | None = None,
    pingpong: bool = False,
) -> list[dict]:
    """
    Build multi-turn chat messages (system, user, assistant) for code-fixing style.
    If pingpong=True: user (task only) -> assistant (one bash) -> user (result) -> ... (no file dump first).
    If pingpong=False: single user (task + code) + single assistant (patch).
    """
    if pingpong:
        return build_messages_pingpong(
            problem_statement, code_context, gold_patch,
            system_prompt=system_prompt, swe_synth_templates=swe_synth_templates,
        )
    if swe_synth_templates:
        system_content = swe_synth_templates["system_template"]
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

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def load_tasks_from_file(input_path: str | Path) -> list[dict]:
    """Load tasks from JSONL (one JSON per line) or JSON array."""
    path = Path(input_path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if not raw:
        return []
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if all(ln.startswith("{") for ln in lines):
        return [json.loads(ln) for ln in lines]
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Convert task -> messages (simple or with code context)
# ---------------------------------------------------------------------------


def task_to_messages_simple(
    instance: dict,
    system_prompt: str | None = None,
    swe_synth_templates: dict | None = None,
    task_id: int | str | None = None,
) -> dict | None:
    """Build messages for one task without code context (no repo clone)."""
    fields = get_task_fields(instance)
    if fields is None:
        return None
    problem_statement = _normalize_problem_statement(fields["problem_statement"])
    gold_patch = (fields["gold_patch"] or "").strip()
    if not problem_statement or not gold_patch:
        return None
    instance_id = fields.get("instance_id", "unknown")
    # Ping-pong: first user = task only (no file structure); assistant = one step; user = result; etc.
    messages = build_messages(
        problem_statement,
        NO_CODE_PLACEHOLDER,
        gold_patch,
        system_prompt=system_prompt,
        swe_synth_templates=swe_synth_templates,
        pingpong=True,
    )
    return {
        "instance_id": instance_id,
        "task_id": task_id if task_id is not None else instance_id,
        "messages": messages,
    }


def task_to_messages_with_code(
    instance: dict,
    repo_cache_dir: str | Path,
    system_prompt: str | None = None,
    swe_synth_templates: dict | None = None,
    task_id: int | str | None = None,
    clone_url_template: str = "https://github.com/{repo}.git",
) -> dict | None:
    """Build messages for one task with code context (clone repo, read files at base_commit)."""
    if Repo is None:
        raise RuntimeError("Install gitpython: pip install gitpython")
    fields = get_task_fields(instance)
    if fields is None:
        return None
    repo_name = fields["repo"]
    base_commit = fields["base_commit"]
    problem_statement = _normalize_problem_statement(fields["problem_statement"])
    gold_patch = (fields["gold_patch"] or "").strip()
    instance_id = fields.get("instance_id", "unknown")
    if not problem_statement or not gold_patch:
        return None

    repo_dir = os.path.join(repo_cache_dir, repo_name.replace("/", "_"))
    git_dir = os.path.join(repo_dir, ".git")
    if os.path.isdir(git_dir):
        LOG.debug("Using cached repo: %s", repo_dir)
    else:
        if os.path.exists(repo_dir):
            LOG.warning("Removing non-git dir and re-cloning: %s", repo_dir)
            shutil.rmtree(repo_dir, ignore_errors=True)
        url = clone_url_template.format(repo=repo_name)
        LOG.info("Cloning %s -> %s", url, repo_dir)
        Repo.clone_from(url, repo_dir)

    modified_files = get_modified_files(gold_patch)
    code_parts = []
    for fpath in modified_files:
        content = load_file_at_commit(repo_dir, fpath, base_commit)
        if content is None:
            continue
        code_parts.append(f"### File: {fpath}\n{content}")
    code_context = "\n".join(code_parts) if code_parts else NO_CODE_PLACEHOLDER

    # Ping-pong: first user = task only (no file structure); then assistant (cat file) -> user (output) -> assistant (patch) -> user (result) -> assistant (complete)
    messages = build_messages(
        problem_statement,
        code_context,
        gold_patch,
        system_prompt=system_prompt,
        swe_synth_templates=swe_synth_templates,
        pingpong=True,
    )
    return {
        "instance_id": instance_id,
        "task_id": task_id if task_id is not None else instance_id,
        "messages": messages,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate multi-turn code-fixing dataset (JSON with messages)"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Input JSON/JSONL when --source file (required if source=file)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output JSON file (default: data/swe_dataset_messages.json)",
    )
    parser.add_argument(
        "--mode",
        choices=("simple", "code-context"),
        default="code-context",
        help="simple = issue+patch only; code-context = issue+current code+patch (default: code-context)",
    )
    parser.add_argument(
        "--source",
        choices=("file", "r2", "local"),
        default="file",
        help="Task source (default: file)",
    )
    parser.add_argument(
        "--task-range",
        type=str,
        default="0-99",
        metavar="START-END",
        help="Task ID range when source is r2 or local (default: 0-99)",
    )
    parser.add_argument(
        "--r2-url",
        type=str,
        default=DEFAULT_R2_URL,
        help="R2 base URL when source=r2",
    )
    parser.add_argument(
        "--r2-prefix",
        type=str,
        default=DEFAULT_R2_PREFIX,
        help="R2 prefix when source=r2",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        default=Path("/tmp/swe-synth-cache"),
        help="Local task cache when source=local",
    )
    parser.add_argument(
        "--repo-cache",
        type=Path,
        default=None,
        help="Git repo cache for code-context (default: <project>/data/repo_cache)",
    )
    parser.add_argument(
        "--swe-synth-config",
        type=Path,
        default=None,
        help="SWE-SYNTH agent config YAML (default: config/swe_synth_agent.yaml)",
    )
    parser.add_argument(
        "--no-swe-synth-format",
        action="store_true",
        help="Use simple Issue/Current code/Patch format instead of SWE-SYNTH templates",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Override system prompt (ignored if SWE-SYNTH config is used)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.3,
        metavar="SEC",
        help="Delay between R2 requests (default: 0.3)",
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

    out_path = args.output or str(ROOT / "data" / "swe_dataset_messages.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    swe_synth_templates = None
    if not args.no_swe_synth_format:
        config_path = args.swe_synth_config or (ROOT / "config" / "swe_synth_agent.yaml")
        swe_synth_templates = load_swe_synth_templates(config_path)
        if swe_synth_templates:
            LOG.info("Using SWE-SYNTH format from %s", config_path)

    if args.source == "file":
        if not args.input:
            parser.error("Input file required when --source=file")
        tasks = load_tasks_from_file(args.input)
        # (task_id, instance); task_id None when from file
        instance_list = [(None, t) for t in tasks]
        LOG.info("Loaded %s tasks from %s", len(instance_list), args.input)
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
        instance_list = []
        for task_id in range(start, end + 1):
            if args.source == "r2" and load_task_from_r2:
                task = load_task_from_r2(task_id, args.r2_url, args.r2_prefix)
            elif args.source == "local" and load_task_from_local:
                task = load_task_from_local(task_id, args.local_cache)
            else:
                task = None
            if task is not None:
                instance_list.append((task_id, task))
            if args.source == "r2" and task_id < end:
                time.sleep(args.request_delay)
        LOG.info("Loaded %s tasks from %s (range %s-%s)", len(instance_list), args.source, start, end)

    examples = []
    repo_cache = args.repo_cache or (ROOT / "data" / "repo_cache")
    if args.mode == "code-context":
        os.makedirs(repo_cache, exist_ok=True)

    for task_id, inst in instance_list:
        try:
            if args.mode == "simple":
                row = task_to_messages_simple(
                    inst,
                    system_prompt=args.system_prompt,
                    swe_synth_templates=swe_synth_templates,
                    task_id=task_id,
                )
            else:
                row = task_to_messages_with_code(
                    inst,
                    str(repo_cache),
                    system_prompt=args.system_prompt,
                    swe_synth_templates=swe_synth_templates,
                    task_id=task_id,
                )
            if row is not None:
                examples.append(row)
        except Exception as e:
            fid = (get_task_fields(inst) or {}).get("instance_id", "?")
            LOG.warning("Skipping %s: %s", fid, e)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    LOG.info("Wrote %s examples to %s", len(examples), out_path)


if __name__ == "__main__":
    main()
