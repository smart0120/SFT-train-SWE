"""
Generate SFT dataset from SWE-SYNTH tasks.

Uses the same task format as affinetes/environments/SWE-SYNTH:
- task / eval = problem_statement (instruction)
- correct answer = gold_patch (response)

Loads tasks from:
1. Public R2 CDN (default), or
2. Local cache directory (e.g. from running SWE-SYNTH eval).

Output: JSON lines or JSON array with {"instruction": problem_statement, "response": gold_patch}.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

# Default public R2 (read-only, no auth)
DEFAULT_R2_URL = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev"
DEFAULT_R2_PREFIX = "bugs"


def load_task_from_r2(task_id: int, base_url: str, prefix: str, timeout: int = 30) -> dict | None:
    """Load a single task JSON from R2 by task_id."""
    if not httpx:
        raise RuntimeError("Install httpx: pip install httpx")
    key = f"{prefix}/task_{task_id:011d}.json"
    url = f"{base_url.rstrip('/')}/{key}"
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code == 200 and r.content:
            return r.json()
        return None
    except Exception as e:
        print(f"Warning: task {task_id} failed: {e}", file=sys.stderr)
        return None


def load_task_from_local(task_id: int, cache_dir: Path) -> dict | None:
    """Load a single task JSON from local cache (same layout as SWE-SYNTH cache)."""
    path = cache_dir / f"task_{task_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def task_to_instruction_response(task: dict) -> dict | None:
    """
    Convert SWE-SYNTH task (BreakerOutput-like dict) to SFT example.

    - instruction: problem_statement (the task / eval description)
    - response: gold_patch (the correct fix)
    """
    bug = task.get("bug", {})
    original = task.get("original", {})
    problem_statement = bug.get("problem_statement", "").strip()
    gold_patch = original.get("gold_patch", "").strip()
    if not problem_statement or not gold_patch:
        return None
    return {
        "instruction": problem_statement,
        "response": gold_patch,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset from SWE-SYNTH tasks (task=problem_statement, correct answer=gold_patch)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/swe_synth_train.json"),
        help="Output JSON file (default: data/swe_synth_train.json)",
    )
    parser.add_argument(
        "--task-range",
        type=str,
        default="0-99",
        metavar="START-END",
        help="Task ID range inclusive, e.g. 0-99 or 10-199 (default: 0-99)",
    )
    parser.add_argument(
        "--source",
        choices=("r2", "local"),
        default="r2",
        help="Source: r2 (public CDN) or local cache (default: r2)",
    )
    parser.add_argument(
        "--r2-url",
        type=str,
        default=DEFAULT_R2_URL,
        help=f"R2 base URL (default: {DEFAULT_R2_URL})",
    )
    parser.add_argument(
        "--r2-prefix",
        type=str,
        default=DEFAULT_R2_PREFIX,
        help=f"R2 key prefix (default: {DEFAULT_R2_PREFIX})",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        default=Path("/tmp/swe-synth-cache"),
        help="Local cache dir when --source local (default: /tmp/swe-synth-cache)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write JSONL (one JSON object per line) instead of a single JSON array",
    )
    args = parser.parse_args()

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

    examples = []
    for task_id in range(start, end + 1):
        if args.source == "r2":
            task = load_task_from_r2(task_id, args.r2_url, args.r2_prefix)
        else:
            task = load_task_from_local(task_id, args.local_cache)
        if task is None:
            continue
        ex = task_to_instruction_response(task)
        if ex is None:
            continue
        ex["task_id"] = task_id
        examples.append(ex)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        with open(args.output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(examples)} examples to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
