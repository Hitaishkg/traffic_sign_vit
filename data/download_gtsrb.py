"""
Download GTSRB dataset to data/raw/.

Uses torchvision.datasets.GTSRB which fetches from the official GTSRB source.
Re-run safe: torchvision skips download if files already exist and checksums match.
"""

import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import torchvision.datasets as tv_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths relative to this file: data/raw/
RAW_DIR = Path(__file__).parent / "raw"
NUM_CLASSES = 43
TRAIN_FOLDER = RAW_DIR / "gtsrb" / "GTSRB" / "Training"
TEST_FOLDER = RAW_DIR / "gtsrb" / "GTSRB" / "Final_Test" / "Images"


def _download_split(root: Path, split: str) -> tv_datasets.GTSRB:
    """Download a single GTSRB split. Returns the dataset object."""
    logger.info("Checking %s split ...", split)
    dataset = tv_datasets.GTSRB(
        root=str(root),
        split=split,
        download=True,
    )
    logger.info("%s split ready: %d samples", split, len(dataset))
    return dataset


def _verify_train_structure(train_folder: Path) -> Optional[str]:
    """
    Verify train split has exactly NUM_CLASSES subdirectories.
    Returns an error string if verification fails, None if OK.
    """
    if not train_folder.exists():
        return f"Train folder not found: {train_folder}"
    class_dirs = [d for d in train_folder.iterdir() if d.is_dir()]
    if len(class_dirs) != NUM_CLASSES:
        return (
            f"Expected {NUM_CLASSES} class folders, found {len(class_dirs)}. "
            "Re-run with a clean data/raw/ directory."
        )
    return None


def _class_counts_from_disk(train_folder: Path) -> Counter:
    """Count images per class by reading folder structure. GTSRB images are .ppm format."""
    counts: Counter = Counter()
    for class_dir in sorted(train_folder.iterdir()):
        if class_dir.is_dir():
            class_id = int(class_dir.name)
            counts[class_id] = sum(1 for _ in class_dir.glob("*.ppm"))
    return counts


def _print_summary(
    train_ds: tv_datasets.GTSRB,
    test_ds: tv_datasets.GTSRB,
    counts: Counter,
) -> None:
    """Print dataset summary to stdout."""
    total = len(train_ds) + len(test_ds)
    most_common = counts.most_common(3)
    least_common = counts.most_common()[-3:]

    print("\n" + "=" * 55)
    print("  GTSRB Dataset Summary")
    print("=" * 55)
    print(f"  Train samples   : {len(train_ds):>7,}")
    print(f"  Test samples    : {len(test_ds):>7,}")
    print(f"  Total samples   : {total:>7,}")
    print(f"  Classes         : {NUM_CLASSES}")
    print(f"  Location        : {RAW_DIR.resolve()}")
    print("-" * 55)
    print("  Top 3 classes by sample count:")
    for class_id, count in most_common:
        print(f"    Class {class_id:02d}  ->  {count:,} images")
    print("  Bottom 3 classes by sample count:")
    for class_id, count in least_common:
        print(f"    Class {class_id:02d}  ->  {count:,} images")
    print("=" * 55 + "\n")


def download_gtsrb(root: Path = RAW_DIR) -> None:
    """Download GTSRB train and test splits to root/."""
    root.mkdir(parents=True, exist_ok=True)

    try:
        train_ds = _download_split(root, split="train")
        test_ds = _download_split(root, split="test")
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    error = _verify_train_structure(TRAIN_FOLDER)
    if error:
        logger.error("Verification failed: %s", error)
        sys.exit(1)

    counts = _class_counts_from_disk(TRAIN_FOLDER)
    _print_summary(train_ds, test_ds, counts)


if __name__ == "__main__":
    download_gtsrb()
