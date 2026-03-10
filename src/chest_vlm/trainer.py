from __future__ import annotations

import json
from pathlib import Path

from peft import LoraConfig, get_peft_model
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments

from chest_vlm.config import TrainConfig
from chest_vlm.data import ChestXrayInstructionDataset, DataCollatorForVisionLanguage
from chest_vlm.utils import seed_everything


def _build_model_and_processor(config: TrainConfig):
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    return model, processor


def _build_trainer(config: TrainConfig) -> Trainer:
    model, processor = _build_model_and_processor(config)
    train_dataset = ChestXrayInstructionDataset(
        manifest_path=config.train_manifest,
        processor=processor,
        default_prompt=config.prompt,
        max_length=config.max_length,
    )
    eval_dataset = None
    if config.val_manifest:
        eval_dataset = ChestXrayInstructionDataset(
            manifest_path=config.val_manifest,
            processor=processor,
            default_prompt=config.prompt,
            max_length=config.max_length,
        )

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=config.eval_steps if eval_dataset is not None else None,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        report_to=[],
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config.seed,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForVisionLanguage(pad_token_id=processor.tokenizer.pad_token_id or 0),
        processing_class=processor,
    )


def run_training(config: TrainConfig) -> None:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer = _build_trainer(config)
    trainer.train()
    trainer.save_model(config.output_dir)

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.__dict__, handle, indent=2, default=lambda value: value.__dict__)
