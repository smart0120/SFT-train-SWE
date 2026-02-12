"""Dataset formatting for messages-based SFT (role/content only, chat template)."""
# SFTTrainer expects a "text" column (one string per example). We build it from
# "messages" via the tokenizer's chat template so the model sees the same format
# it was pretrained on. TRL does not accept raw "messages" without this step.


def format_chat(example: dict, tokenizer) -> dict:
    """
    In-place: set example["text"] from example["messages"] using the tokenizer
    chat template (tokenize=False; trainer tokenizes later). Training = full
    completion, so add_generation_prompt=False. Returns example for dataset.map.
    """
    messages = list(example.get("messages") or [])
    if not messages:
        example["text"] = ""
        return example
    normalized = [
        {"role": m.get("role", "user"), "content": (m.get("content") or "") or ""}
        for m in messages
    ]
    example["text"] = tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=False,
    )
    return example
