"""Load training config from YAML and environment variables."""
from pathlib import Path
import os
import yaml


def _env(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key)
    return v if v else default


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "train_config.yaml"
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Env overrides
    if _env("MODEL_ID"):
        config["model"]["model_id"] = _env("MODEL_ID")
    if _env("ADAPTER_OUTPUT_DIR"):
        config["paths"]["adapter_output_dir"] = _env("ADAPTER_OUTPUT_DIR")
        config["training"]["output_dir"] = _env("ADAPTER_OUTPUT_DIR")
    if _env("MERGED_OUTPUT_DIR"):
        config["paths"]["merged_output_dir"] = _env("MERGED_OUTPUT_DIR")
    if _env("RUN_NAME"):
        config["run_name"] = _env("RUN_NAME")
    if _env("WANDB_PROJECT"):
        config.setdefault("wandb", {})["project"] = _env("WANDB_PROJECT")
    if _env("WANDB_ENTITY"):
        config.setdefault("wandb", {})["entity"] = _env("WANDB_ENTITY")

    return config
