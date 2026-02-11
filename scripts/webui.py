"""
Web UI to chat with a merged (or adapter) model.
- Load model from path (merged dir, e.g. outputs/merged/<run_name>).
- Set system prompt and user message; continue the conversation in the chat.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

_env_file = ROOT / ".env"
if _env_file.exists():
    with open(_env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import gradio as gr

from inference import load_model_for_inference, generate_chat

# Loaded model and tokenizer (set by "Load model" button)
MODEL = None
TOKENIZER = None


def _content_to_text(content) -> str:
    """Normalize message content to string. Gradio 6 uses list of blocks: [{\"type\": \"text\", \"text\": \"...\"}]."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
        return " ".join(parts)
    return str(content)


def build_messages(system_prompt: str, history: list, user_message: str) -> list[dict]:
    """Build messages list: system + history + new user message. Compatible with Gradio 5 (tuple pairs) and Gradio 6 (messages with role/content, content may be list of blocks)."""
    messages = []
    if (system_prompt or "").strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for item in history or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            u, a = item[0], item[1]
            if u:
                messages.append({"role": "user", "content": _content_to_text(u)})
            if a:
                messages.append({"role": "assistant", "content": _content_to_text(a)})
        elif isinstance(item, dict) and "role" in item:
            content = _content_to_text(item.get("content", ""))
            if content:
                messages.append({"role": item["role"], "content": content})
        elif hasattr(item, "role") and hasattr(item, "content"):
            content = _content_to_text(getattr(item, "content", ""))
            if content:
                messages.append({"role": item.role, "content": content})
    messages.append({"role": "user", "content": _content_to_text(user_message)})
    return messages


def chat_fn(
    message: str,
    history: list,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate assistant reply given message, history, and system prompt."""
    if MODEL is None or TOKENIZER is None:
        return "Load a model first (set path and click **Load model**)."
    user_msg = _content_to_text(message) if message is not None else ""
    if not user_msg.strip():
        return "Please enter a message."
    messages = build_messages(system_prompt or "", history, user_msg.strip())
    try:
        reply = generate_chat(
            MODEL,
            TOKENIZER,
            messages,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(0.01, temperature),
        )
        return reply or "(empty reply)"
    except Exception as e:
        return f"Error: {e}"


def load_model_fn(model_path: str) -> str:
    """Load merged model from path. Returns status message."""
    global MODEL, TOKENIZER
    path = (model_path or "").strip()
    if not path:
        return "Enter a model path (e.g. outputs/merged/qwen2.5-4b-lora-h100)."
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = ROOT / path
    if not resolved.exists():
        return f"Path does not exist: {resolved}"
    try:
        MODEL, TOKENIZER = load_model_for_inference(model_path=str(resolved))
        return f"Model loaded: {resolved.name}"
    except Exception as e:
        MODEL, TOKENIZER = None, None
        return f"Load failed: {e}"


def main():
    default_path = str(ROOT / "outputs" / "merged")
    with gr.Blocks() as demo:
        gr.Markdown("## Chat with merged model")
        with gr.Row():
            model_path = gr.Textbox(
                label="Model path (merged dir)",
                placeholder="e.g. outputs/merged/qwen2.5-4b-lora-h100",
                value=default_path,
                scale=4,
            )
            load_btn = gr.Button("Load model", variant="primary", scale=1)
        load_status = gr.Textbox(label="Status", interactive=False)

        system_prompt = gr.Textbox(
            label="System prompt",
            placeholder="You are a helpful assistant.",
            value="You are a helpful assistant.",
            lines=3,
        )
        with gr.Accordion("Generation options", open=False):
            max_new_tokens = gr.Slider(64, 2048, value=512, step=64, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature (0 = greedy)")

        chat = gr.ChatInterface(
            fn=chat_fn,
            additional_inputs=[system_prompt, max_new_tokens, temperature],
        )

        load_btn.click(
            load_model_fn,
            inputs=[model_path],
            outputs=[load_status],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)


if __name__ == "__main__":
    main()
