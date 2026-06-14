"""
GTSRBDataset: PyTorch Dataset wrapping torchvision.datasets.GTSRB.

Splits:
  - "train" : 85% of official train set, stratified by class
  - "val"   : 15% of official train set, stratified by class
  - "test"  : official test set (all 12,630 images, flat folder + CSV)

Torchvision is used as the backing loader for all splits to avoid re-parsing
raw files. The train/val stratified split is applied on top of torchvision's
_samples list, using a fixed seed for reproducibility.
"""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.datasets as tv_datasets

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "raw"
NUM_CLASSES = 43
VALID_SPLITS = ("train", "val", "test")

# Type alias for a single sample entry from torchvision
Sample = Tuple[str, int]


def _stratified_split(
    samples: List[Sample],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Sample], List[Sample]]:
    """
    Split sample list into train/val subsets, stratified by class label.

    Each class contributes `val_fraction` of its samples to val (minimum 1).
    Uses stdlib random with a fixed seed — no numpy or sklearn dependency.

    Args:
        samples: List of (img_path, label) tuples from torchvision GTSRB.
        val_fraction: Fraction of each class to assign to val split.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (train_samples, val_samples).
    """
    class_buckets: Dict[int, List[Sample]] = defaultdict(list)
    for path, label in samples:
        class_buckets[label].append((path, label))

    rng = random.Random(seed)
    train_samples: List[Sample] = []
    val_samples: List[Sample] = []

    for label in sorted(class_buckets.keys()):
        bucket = class_buckets[label][:]  # copy to avoid mutating torchvision internals
        rng.shuffle(bucket)
        n_val = max(1, round(len(bucket) * val_fraction))
        val_samples.extend(bucket[:n_val])
        train_samples.extend(bucket[n_val:])

    return train_samples, val_samples


class GTSRBDataset(Dataset):
    """
    GTSRB dataset with train/val/test split support.

    Args:
        split: One of "train", "val", "test".
        transform: Optional callable applied to each PIL Image before return.
        root: Path to data/raw/. Defaults to data/raw/ relative to this file.
        val_fraction: Fraction of official train set held out for val. Default 0.15.
        seed: RNG seed for the stratified train/val split. Default 42.

    Example:
        >>> from data.transforms import get_train_transform, get_eval_transform
        >>> train_ds = GTSRBDataset("train", transform=get_train_transform())
        >>> val_ds   = GTSRBDataset("val",   transform=get_eval_transform())
        >>> test_ds  = GTSRBDataset("test",  transform=get_eval_transform())
    """

    NUM_CLASSES: int = NUM_CLASSES

    def __init__(
        self,
        split: str,
        transform: Optional[Callable] = None,
        root: Path = RAW_DIR,
        val_fraction: float = 0.15,
        seed: int = 42,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got '{split}'")

        self.split = split
        self.transform = transform

        if split in ("train", "val"):
            backing = tv_datasets.GTSRB(root=str(root), split="train", download=False)
            if not hasattr(backing, "_samples"):
                raise RuntimeError(
                    "torchvision.datasets.GTSRB does not expose '_samples'. "
                    "Check torchvision version compatibility."
                )
            train_samples, val_samples = _stratified_split(
                backing._samples, val_fraction, seed
            )
            self._samples: List[Sample] = (
                train_samples if split == "train" else val_samples
            )
        else:  # test
            backing = tv_datasets.GTSRB(root=str(root), split="test", download=False)
            if not hasattr(backing, "_samples"):
                raise RuntimeError(
                    "torchvision.datasets.GTSRB does not expose '_samples'. "
                    "Check torchvision version compatibility."
                )
            self._samples = backing._samples

        logger.info(
            "GTSRBDataset | split=%-5s | samples=%d", split, len(self._samples)
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx: Sample index.

        Returns:
            Tuple of (image_tensor, class_label).
            image_tensor shape depends on the transform applied.
            class_label is an int in [0, 42].
        """
        img_path, label = self._samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def class_counts(self) -> Dict[int, int]:
        """
        Return per-class sample counts for this split.

        Useful for computing class weights for weighted loss or sampler.

        Returns:
            Dict mapping class_id -> count, sorted by class_id.
        """
        counts: Dict[int, int] = defaultdict(int)
        for _, label in self._samples:
            counts[label] += 1
        return dict(sorted(counts.items()))
