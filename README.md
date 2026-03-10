# Chest X-Ray Vision-Language Fine-Tuning

This repository contains a minimal project for fine-tuning a vision-language model (VLM) on chest X-ray images paired with medical explanations.

The default setup uses:

- Hugging Face `transformers`
- Parameter-efficient fine-tuning with `peft` LoRA
- JSONL manifests for train/validation/test data
- Single-image supervised instruction tuning

## Use Case

Given a chest X-ray image, the model is trained to answer a clinical prompt such as:

`Analyze this chest X-ray and provide a concise medical explanation of the findings.`

The target text should be a medically grounded explanation, not just a class label.

## Repository Layout

```text
.
├── configs/
│   └── train_config.yaml
├── data/
│   └── sample_manifest.jsonl
├── requirements.txt
├── scripts/
│   ├── evaluate.py
│   ├── infer.py
│   └── train.py
└── src/
    └── chest_vlm/
        ├── __init__.py
        ├── config.py
        ├── data.py
        ├── prompts.py
        ├── trainer.py
        └── utils.py
```

## Dataset Format

Use JSONL where each row contains at least:

```json
{
  "image": "images/patient_001.png",
  "explanation": "Portable AP chest radiograph shows bibasilar patchy opacities, greater on the right, with no pleural effusion or pneumothorax. Findings are suspicious for multifocal pneumonia.",
  "prompt": "Analyze this chest X-ray and provide a medical explanation."
}
```

Fields:

- `image`: path to image file, absolute or relative to the manifest
- `explanation`: training target
- `prompt`: optional instruction override per sample
- `id`: optional stable identifier

## Recommended Data Sources

For a real project, use a properly licensed radiology dataset with image-report pairs, such as:

- MIMIC-CXR with appropriate credentialing
- OpenI IU X-Ray
- CheXpert-derived report datasets where licensing and usage terms fit your work

You are responsible for compliance, privacy, and clinical governance.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Configure Training

Edit [configs/train_config.yaml](/workspace/configs/train_config.yaml) and set:

- `model_name`
- dataset manifest paths
- batch size
- learning rate
- output directory

The default model is a placeholder VLM that works with `AutoProcessor` and `AutoModelForVision2Seq`. Choose a model that supports image-text generation in your environment.

Examples that typically fit this pattern:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `HuggingFaceM4/idefics2-8b`

Check the model card before use because processor behavior and memory needs differ.

## Train

```bash
python scripts/train.py --config configs/train_config.yaml
```

## Inference

```bash
python scripts/infer.py --model-path outputs/chest-vlm --image path/to/xray.png
```

## Evaluate

```bash
python scripts/evaluate.py --predictions outputs/predictions.jsonl --references data/val.jsonl
```

The evaluation script computes lightweight text overlap metrics intended for debugging, not clinical validation.

## Notes

- This project is a starting point, not a validated medical device workflow.
- Use a radiologist-reviewed evaluation protocol before any real deployment.
- Generated explanations may hallucinate findings. Human oversight is required.
