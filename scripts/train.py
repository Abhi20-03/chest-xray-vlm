from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from chest_vlm.config import load_config
from chest_vlm.trainer import run_training


def main() -> None:
    config = load_config()
    run_training(config)


if __name__ == "__main__":
    main()
