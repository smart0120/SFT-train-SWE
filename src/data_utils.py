"""Dataset formatting for messages-based SFT (role/content only, chat template)."""
# No dataset loading here; train.py loads JSON and expects "messages" column.


def messages_to_text(example: dict, tokenizer) -> str:
    """
    Convert a single example with a "messages" field to training text using only
    the tokenizer's chat template. Only role and content are used.
    """
    messages = list(example.get("messages") or [])
    if not messages:
        return ""
    normalized = [
        {"role": m.get("role", "user"), "content": (m.get("content") or "") or ""}
        for m in messages
    ]
    return tokenizer.apply_chat_template(
        normalized,
        tokenize=False,
        add_generation_prompt=False,
    )
