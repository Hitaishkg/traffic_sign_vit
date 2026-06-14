"""
Block 1 — EDA: GTSRB Dataset Exploration

Run from project root:
    python notebooks/01_eda.py

Outputs:
    - Class distribution (top 5 most + least common) printed to stdout
    - Min/max image dimensions printed to stdout
    - 5x5 random sample grid saved to notebooks/outputs/eda_grid.png
"""

import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Allow imports from project root regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GTSRBDataset, NUM_CLASSES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Official GTSRB class names (43 classes, 0-indexed)
CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing (>3.5t)",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles >3.5t prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing (>3.5t)",
}


def print_class_distribution(counts: Dict[int, int], top_n: int = 5) -> None:
    """Print top N most and least common classes by sample count."""
    sorted_by_count = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print("\n" + "=" * 60)
    print(f"  Class Distribution (train split, {sum(counts.values()):,} total samples)")
    print("=" * 60)

    print(f"\n  Top {top_n} most common:")
    for class_id, count in sorted_by_count[:top_n]:
        bar = "#" * (count // 50)
        print(f"    [{class_id:02d}] {CLASS_NAMES[class_id]:<35}  {count:>5,}  {bar}")

    print(f"\n  Top {top_n} least common:")
    for class_id, count in sorted_by_count[-top_n:]:
        bar = "#" * (count // 50)
        print(f"    [{class_id:02d}] {CLASS_NAMES[class_id]:<35}  {count:>5,}  {bar}")

    print("\n" + "=" * 60)


def print_dimension_stats(
    samples: List[Tuple[str, int]], n_sample: int = 500
) -> None:
    """
    Print min/max image dimensions from a random sample of images.

    Samples a subset rather than iterating all 39k to keep EDA fast.
    PPM header reads are quick but still O(n) at scale.
    """
    rng = random.Random(SEED)
    subset = rng.sample(samples, min(n_sample, len(samples)))

    widths, heights = [], []
    for img_path, _ in subset:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    print("\n  Image Dimension Stats (sampled from train, n={})".format(len(subset)))
    print(f"    Width  — min: {min(widths)}px  max: {max(widths)}px  avg: {int(np.mean(widths))}px")
    print(f"    Height — min: {min(heights)}px  max: {max(heights)}px  avg: {int(np.mean(heights))}px")
    print()


def plot_sample_grid(
    samples: List[Tuple[str, int]],
    grid_size: int = 5,
    output_path: Path = OUTPUT_DIR / "eda_grid.png",
) -> None:
    """
    Plot a grid_size x grid_size grid of random training images with class names.

    Images are shown at native resolution (no normalization) so colours are true.
    """
    rng = random.Random(SEED)
    selected = rng.sample(samples, grid_size * grid_size)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 14))
    fig.suptitle("GTSRB — Random Sample Grid (train split)", fontsize=14, y=1.01)

    for ax, (img_path, label) in zip(axes.flat, selected):
        img = Image.open(img_path).convert("RGB")
        ax.imshow(img)
        ax.set_title(
            f"[{label:02d}] {CLASS_NAMES[label]}",
            fontsize=6,
            pad=3,
        )
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Grid saved to %s", output_path.resolve())

    try:
        plt.show()
    except Exception:
        # Non-interactive environment (headless server, CI) — skip display
        pass

    plt.close(fig)


def main() -> None:
    logger.info("Loading train split (no transforms) ...")
    # No transform — we want raw PIL images for true colours and native dimensions
    train_ds = GTSRBDataset(split="train", transform=None)

    # --- Class distribution ---
    counts = train_ds.class_counts()
    print_class_distribution(counts, top_n=5)

    # --- Image dimension stats ---
    print_dimension_stats(train_ds._samples, n_sample=500)

    # --- Sample grid ---
    logger.info("Generating 5x5 sample grid ...")
    plot_sample_grid(train_ds._samples, grid_size=5)

    logger.info("EDA complete.")


if __name__ == "__main__":
    main()
