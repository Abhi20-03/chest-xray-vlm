from __future__ import annotations

from dataclasses import dataclass, field
import argparse
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraConfigSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    model_name: str
    train_manifest: str
    val_manifest: str | None = None
    test_manifest: str | None = None
    output_dir: str = "outputs/chest-vlm"
    prompt: str = "Analyze this chest X-ray and provide a concise medical explanation of the findings."
    max_length: int = 768
    max_new_tokens: int = 192
    image_size: int = 448
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    lora: LoraConfigSpec = field(default_factory=LoraConfigSpec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a chest X-ray vision-language model.")
    parser.add_argument("--config", required=True, help="Path to YAML configuration.")
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return data


def _resolve_path(base_dir: Path, value: str | None) -> str | None:
    if not value:
        return value
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((base_dir / candidate).resolve())


def load_config() -> TrainConfig:
    args = parse_args()
    config_path = Path(args.config).resolve()
    raw = _load_yaml(config_path)
    raw_lora = raw.get("lora", {})
    raw["lora"] = LoraConfigSpec(**raw_lora)
    config = TrainConfig(**raw)
    base_dir = config_path.parent
    config.train_manifest = _resolve_path(base_dir, config.train_manifest)
    config.val_manifest = _resolve_path(base_dir, config.val_manifest)
    config.test_manifest = _resolve_path(base_dir, config.test_manifest)
    config.output_dir = _resolve_path(base_dir, config.output_dir) or config.output_dir
    return config
