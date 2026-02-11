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
import logging
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None

# Default public R2 (read-only, no auth)
DEFAULT_R2_URL = "https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev"
DEFAULT_R2_PREFIX = "bugs"

# Logging
LOG = logging.getLogger(__name__)


def load_task_from_r2(
    task_id: int,
    base_url: str,
    prefix: str,
    timeout: int = 30,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> dict | None:
    """Load a single task JSON from R2 by task_id, with retries and backoff."""
    if not httpx:
        raise RuntimeError("Install httpx: pip install httpx")
    key = f"{prefix}/task_{task_id:011d}.json"
    url = f"{base_url.rstrip('/')}/{key}"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            r = httpx.get(url, timeout=timeout)
            if r.status_code == 200 and r.content:
                return r.json()
            last_error = f"HTTP {r.status_code}"
        except Exception as e:
            last_error = e
        if attempt < max_retries:
            LOG.warning("Task %s attempt %s/%s failed (%s), retrying in %.1fs ...", task_id, attempt, max_retries, last_error, retry_delay)
            time.sleep(retry_delay)
    LOG.warning("Task %s failed after %s attempts: %s", task_id, max_retries, last_error)
    return None


def load_task_from_local(task_id: int, cache_dir: Path) -> dict | None:
    """Load a single task JSON from local cache (same layout as SWE-SYNTH cache)."""
    path = cache_dir / f"task_{task_id}.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_instruction(text: str) -> str:
    """Strip whitespace and optional surrounding double quotes from problem_statement."""
    s = (text or "").strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def task_to_instruction_response(task: dict, task_id: int | None = None) -> dict | None:
    """
    Convert SWE-SYNTH task (BreakerOutput-like dict) to SFT example for swe_synth_train.json.

    Output format: {"instruction": problem_statement, "response": gold_patch, "task_id": task_id}
    - instruction: problem_statement (user-facing task / bug description), normalized
    - response: gold_patch (unified diff, the correct fix)
    - task_id: optional, for tracing back to the source task
    """
    bug = task.get("bug", {})
    original = task.get("original", {})
    problem_statement = _normalize_instruction(bug.get("problem_statement", "") or "")
    gold_patch = (original.get("gold_patch", "") or "").strip()
    if not problem_statement or not gold_patch:
        return None
    out = {
        "instruction": problem_statement,
        "response": gold_patch,
    }
    if task_id is not None:
        out["task_id"] = task_id
    return out


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
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        metavar="N",
        help="Max retries per task when fetching from R2 (default: 3)",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Seconds to wait between retries (default: 2.0)",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.5,
        metavar="SEC",
        help="Seconds to wait between requests (avoid rate limit; default: 0.5)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging (INFO level)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

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

    total_tasks = end - start + 1
    LOG.info("Task range: %s-%s (%s tasks), source=%s", start, end, total_tasks, args.source)
    if args.source == "r2":
        LOG.info("Retries=%s, retry_delay=%.1fs, request_delay=%.1fs", args.retries, args.retry_delay, args.request_delay)

    examples = []
    for i, task_id in enumerate(range(start, end + 1), 1):
        LOG.info("Fetching task %s (%s/%s) ...", task_id, i, total_tasks)
        if args.source == "r2":
            task = load_task_from_r2(
                task_id,
                args.r2_url,
                args.r2_prefix,
                max_retries=args.retries,
                retry_delay=args.retry_delay,
            )
            if task_id > start:
                time.sleep(args.request_delay)
        else:
            task = load_task_from_local(task_id, args.local_cache)
        if task is None:
            LOG.debug("Task %s: no data", task_id)
            continue
        ex = task_to_instruction_response(task, task_id=task_id)
        if ex is None:
            LOG.debug("Task %s: skipped (missing problem_statement or gold_patch)", task_id)
            continue
        examples.append(ex)
        LOG.info("Task %s: ok (examples so far: %s)", task_id, len(examples))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        with open(args.output, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)

    LOG.info("Done. Wrote %s examples to %s", len(examples), args.output)


if __name__ == "__main__":
    main()
