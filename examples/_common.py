"""Run any demo:  python examples/01_filters_demo.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "examples" / "outputs"


def save(array, name):
    try:
        from PIL import Image
        from cv_examples.utils import scale_to_uint8

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(scale_to_uint8(array)).save(OUTPUT_DIR / name)
        print(f"saved {name}")
    except ImportError:
        print(f"(Pillow not installed; skipped saving {name})")
