"""
Image transforms for GTSRB training and evaluation.

Uses torchvision.transforms.v2 (modern API, replaces v1).
Provides separate train (with augmentation) and eval (deterministic) pipelines.

Normalization constants:
  - VIT_MEAN/VIT_STD       : for google/vit-base-patch16-224 (pretraining stats)
  - IMAGENET_MEAN/STD      : for ResNet50 baseline (standard ImageNet stats)

Pass the appropriate mean/std when constructing transforms for each model.
"""

from typing import List

import torch
import torchvision.transforms.v2 as v2

# google/vit-base-patch16-224 was pretrained with these normalization values.
# Do NOT substitute ImageNet stats here for ViT — the model expects [0.5, 0.5, 0.5].
VIT_MEAN: List[float] = [0.5, 0.5, 0.5]
VIT_STD: List[float] = [0.5, 0.5, 0.5]

# Standard ImageNet normalization for ResNet50 baseline.
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

IMG_SIZE: int = 224  # ViT-base-patch16-224 expects 224x224 input


def get_train_transform(
    mean: List[float] = VIT_MEAN,
    std: List[float] = VIT_STD,
    img_size: int = IMG_SIZE,
) -> v2.Compose:
    """
    Training transforms with traffic-sign-safe augmentations.

    Augmentation rationale:
      - No horizontal flip: left/right turn signs are different classes (33 vs 34).
        Flipping creates a correctly-shaped but mislabeled sample.
      - RandomAffine (translate + rotate): simulates camera angle and positioning
        variation. Rotation capped at 15 deg — beyond that, sign shape becomes
        ambiguous. Translation at 10% of image size.
      - RandomResizedCrop: simulates distance variation. Scale floor 0.8 preserves
        critical sign content (symbols/text near edges).
      - ColorJitter: simulates lighting variation (overcast, golden hour, night).
        Kept moderate — traffic signs have strict color semantics (red = danger, etc).

    Args:
        mean: Normalization mean per channel.
        std: Normalization std per channel.
        img_size: Output spatial size in pixels. Default 224.

    Returns:
        Composed transform: PIL Image -> normalized float32 tensor (C, H, W).
    """
    return v2.Compose(
        [
            v2.ToImage(),  # PIL -> uint8 ImageTensor
            v2.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
            v2.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0,255] -> float32 [0.0,1.0]
            v2.Normalize(mean=mean, std=std),
        ]
    )


def get_eval_transform(
    mean: List[float] = VIT_MEAN,
    std: List[float] = VIT_STD,
    img_size: int = IMG_SIZE,
) -> v2.Compose:
    """
    Evaluation/test transforms. Fully deterministic — no randomness.

    Resize to 256 then CenterCrop to 224 follows the standard ViT eval protocol
    from the original paper (Dosovitskiy et al. 2020). The 256 -> 224 ratio
    ensures the crop sees the central ~77% of the resized image, avoiding hard
    edges from the resize step.

    Args:
        mean: Normalization mean per channel.
        std: Normalization std per channel.
        img_size: Output spatial size in pixels. Default 224.

    Returns:
        Composed transform: PIL Image -> normalized float32 tensor (C, H, W).
    """
    resize_size = int(img_size * 256 / 224)  # 256 when img_size=224
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(resize_size, antialias=True),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
