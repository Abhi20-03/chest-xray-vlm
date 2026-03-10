from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from PIL import Image
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from chest_vlm.prompts import build_messages
from chest_vlm.utils import extract_assistant_text, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run chest X-ray VLM inference.")
    parser.add_argument("--model-path", required=True, help="Model checkpoint or HF repo ID.")
    parser.add_argument("--image", required=True, help="Path to the chest X-ray image.")
    parser.add_argument(
        "--prompt",
        default="Analyze this chest X-ray and provide a concise medical explanation of the findings.",
        help="Instruction prompt.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    ).to(args.device)
    model.eval()

    image = load_image(args.image)
    messages = build_messages(prompt=args.prompt)
    text = processor.apply_chat_template(messages, add_generation_prompt=True)

    inputs = processor(images=[image], text=[text], return_tensors="pt")
    inputs = {name: tensor.to(args.device) for name, tensor in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)

    prompt_length = inputs["input_ids"].shape[1]
    output_tokens = generated[:, prompt_length:]
    decoded = processor.batch_decode(output_tokens, skip_special_tokens=True)[0].strip()
    result = {"image": str(Path(args.image).resolve()), "prediction": extract_assistant_text(decoded)}

    if args.json:
        print(json.dumps(result, ensure_ascii=True))
    else:
        print(result["prediction"])


if __name__ == "__main__":
    main()
