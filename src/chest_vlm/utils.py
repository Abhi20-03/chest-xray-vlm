from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image
from rouge_score import rouge_scorer
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def extract_assistant_text(text: str) -> str:
    return text.strip()


def compute_text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for prediction, reference in zip(predictions, references):
        scores = scorer.score(reference, prediction)
        for key in rouge_totals:
            rouge_totals[key] += scores[key].fmeasure

    smoothing = SmoothingFunction().method1
    bleu = corpus_bleu(
        list_of_references=[[reference.split()] for reference in references],
        hypotheses=[prediction.split() for prediction in predictions],
        smoothing_function=smoothing,
    )
    count = max(len(predictions), 1)
    return {
        "rouge1": float(rouge_totals["rouge1"] / count),
        "rouge2": float(rouge_totals["rouge2"] / count),
        "rougeL": float(rouge_totals["rougeL"] / count),
        "bleu": float(bleu),
        "sample_count": float(len(predictions)),
    }
