"""Dataset loading and formatting for instruction tuning."""
from datasets import load_dataset, Dataset


def _format_response_agent_style(response: str) -> str:
    """Wrap a raw response (e.g. gold patch) as THOUGHT + single bash block (SWE-SYNTH agent style)."""
    if not response or not response.strip():
        return response
    # Escape the patch for use inside a heredoc
    return (
        "THOUGHT: I will fix the issue by applying the following patch.\n\n"
        "```bash\n"
        "cat << 'PATCH' | git apply -\n"
        + response.strip()
        + "\nPATCH\n"
        "```"
    )


def format_instruction(
    example: dict,
    tokenizer,
    system_prompt: str = "You are a helpful assistant.",
    instance_template: str | None = None,
    format_response_agent_style: bool = False,
) -> str:
    """
    Format a single example into a chat prompt using the tokenizer's chat template.

    - If example has "messages" (list of {"role", "content", ...}): use for multi-turn. Only role and
      content are used; extra fields (e.g. timestamp) are ignored. instance_id/task_id are ignored.
    - Else use instruction/response: user = instance_template with {{task}}=instruction, or plain instruction;
      assistant = response (optionally wrapped in THOUGHT + bash).
    """
    if "messages" in example and example["messages"]:
        # Normalize to only role + content (ignore timestamp and other extra fields)
        raw = list(example["messages"])
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "") or ""}
            for m in raw
        ]
        # Keep system message from example (e.g. SWE-SYNTH template); only prepend if missing
        if messages[0].get("role") != "system" and system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
    else:
        instruction = example.get("instruction", "")
        response = example.get("response", "")
        if format_response_agent_style and response:
            response = _format_response_agent_style(response)
        user_content = instruction
        if instance_template:
            user_content = instance_template.replace("{{task}}", instruction)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": response},
        ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def load_and_prepare_dataset(
    dataset_name: str,
    split: str = "train",
    instruction_column: str = "instruction",
    response_column: str = "response",
    dataset_path: str | None = None,
    data_files: str | list | None = None,
) -> Dataset:
    """
    Load dataset from Hugging Face or local files.
    - dataset_name: HF dataset id or "json"/"csv" for local.
    - dataset_path: optional local path.
    - data_files: for load_dataset(..., data_files=...) when using local files.
    """
    if data_files is not None:
        dataset = load_dataset("json", data_files=data_files, split=split)
    elif dataset_path:
        path = str(dataset_path)
        if path.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=path, split="train")
        elif path.endswith(".json"):
            dataset = load_dataset("json", data_files=path, split="train")
        else:
            dataset = load_dataset(dataset_name or "json", path=path, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)

    # Rename columns if needed (skip if dataset uses "messages" for multi-turn)
    if "messages" not in dataset.column_names:
        if instruction_column != "instruction" and instruction_column in dataset.column_names:
            dataset = dataset.rename_column(instruction_column, "instruction")
        if response_column != "response" and response_column in dataset.column_names:
            dataset = dataset.rename_column(response_column, "response")

    return dataset
