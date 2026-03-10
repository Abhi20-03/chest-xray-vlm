from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chest_vlm.utils import compute_text_metrics, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated explanations.")
    parser.add_argument("--predictions", required=True, help="JSONL with `id` and `prediction`.")
    parser.add_argument("--references", required=True, help="JSONL with `id` and `explanation`.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_rows = read_jsonl(args.predictions)
    reference_rows = read_jsonl(args.references)
    reference_by_id = {row["id"]: row for row in reference_rows}

    preds = []
    refs = []
    for row in prediction_rows:
        sample_id = row["id"]
        if sample_id not in reference_by_id:
            continue
        preds.append(row["prediction"])
        refs.append(reference_by_id[sample_id]["explanation"])

    metrics = compute_text_metrics(preds, refs)
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
