from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset

from chest_vlm.prompts import build_messages


@dataclass
class Sample:
    sample_id: str
    image_path: str
    prompt: str
    explanation: str


def _resolve_image_path(manifest_path: Path, image_path: str) -> str:
    candidate = Path(image_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((manifest_path.parent / candidate).resolve())


def load_manifest(manifest_path: str, default_prompt: str) -> list[Sample]:
    path = Path(manifest_path).resolve()
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if "image" not in row or "explanation" not in row:
                raise ValueError(f"Manifest row {idx} in {path} must contain `image` and `explanation`.")
            samples.append(
                Sample(
                    sample_id=str(row.get("id", idx)),
                    image_path=_resolve_image_path(path, row["image"]),
                    prompt=row.get("prompt", default_prompt),
                    explanation=row["explanation"].strip(),
                )
            )
    if not samples:
        raise ValueError(f"No training rows found in {path}")
    return samples


class ChestXrayInstructionDataset(Dataset):
    def __init__(self, manifest_path: str, processor: Any, default_prompt: str, max_length: int) -> None:
        self.samples = load_manifest(manifest_path, default_prompt)
        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        prompt_messages = build_messages(sample.prompt)
        target_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": sample.explanation}]}
        ]

        input_text = self.processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
        target_text = self.processor.apply_chat_template(target_messages, add_generation_prompt=False)

        prompt_batch = self.processor(images=[image], text=[input_text], return_tensors="pt")
        target_batch = self.processor(images=[image], text=[target_text], return_tensors="pt")

        input_ids = target_batch["input_ids"][0][: self.max_length]
        attention_mask = target_batch["attention_mask"][0][: self.max_length]
        labels = input_ids.clone()

        prompt_len = min(prompt_batch["input_ids"].shape[1], labels.shape[0])
        labels[:prompt_len] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": target_batch["pixel_values"][0],
            "id": sample.sample_id,
        }

        if "image_grid_thw" in target_batch:
            result["image_grid_thw"] = target_batch["image_grid_thw"][0]

        return result


class DataCollatorForVisionLanguage:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_seq_len = max(item["input_ids"].shape[0] for item in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "pixel_values": [],
        }
        image_grid = []

        for item in features:
            pad_len = max_seq_len - item["input_ids"].shape[0]
            batch["input_ids"].append(torch.nn.functional.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id))
            batch["attention_mask"].append(torch.nn.functional.pad(item["attention_mask"], (0, pad_len), value=0))
            batch["labels"].append(torch.nn.functional.pad(item["labels"], (0, pad_len), value=-100))
            batch["pixel_values"].append(item["pixel_values"])
            if "image_grid_thw" in item:
                image_grid.append(item["image_grid_thw"])

        collated = {name: torch.stack(values) for name, values in batch.items()}
        if image_grid:
            collated["image_grid_thw"] = torch.stack(image_grid)
        return collated
