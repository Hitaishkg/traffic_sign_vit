"""
Block 5 — Evaluation on Held-Out Test Set

Evaluates both ViT-base and ResNet50 on the GTSRB test set (12,630 images).
Neither model has seen these images during training or validation.

Outputs:
  - Comparison table printed to console
  - evaluation/outputs/confusion_matrix_vit.png
  - evaluation/outputs/confusion_matrix_resnet.png
  - evaluation/outputs/per_class_accuracy.png
  - evaluation/outputs/test_results.csv

Run from project root:
    ../.venv/Scripts/python evaluation/evaluate.py
"""

import __main__
import csv
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm
from transformers import ViTConfig, ViTForImageClassification

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── TrainConfig stub ───────────────────────────────────────────────────────────
# Checkpoints were saved with the TrainConfig dataclass from each training script
# (pickled as __main__.TrainConfig). Define a compatible stub here and patch
# __main__ so pickle can deserialise both checkpoints without importing the
# training modules (which would trigger training on import).
@dataclass
class TrainConfig:
    model_name:     str   = ""
    num_classes:    int   = 43
    batch_size:     int   = 32
    num_workers:    int   = 4
    val_fraction:   float = 0.15
    seed:           int   = 42
    lr_backbone:    float = 0.0
    lr_head:        float = 0.0
    weight_decay:   float = 0.0
    num_epochs:     int   = 20
    warmup_epochs:  int   = 2
    use_amp:        bool  = True
    checkpoint_dir: Path  = field(default_factory=lambda: Path("checkpoints"))
    log_csv:        Path  = field(default_factory=lambda: Path("training/logs/metrics.csv"))

__main__.TrainConfig = TrainConfig   # makes pickle happy for both checkpoints

from data.dataset import GTSRBDataset, NUM_CLASSES
from data.transforms import (
    IMAGENET_MEAN, IMAGENET_STD,
    VIT_MEAN, VIT_STD,
    get_eval_transform,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR    = Path(__file__).parent / "outputs"
CHECKPOINT_DIR = Path("checkpoints")
BATCH_SIZE    = 64
NUM_WORKERS   = 4
LATENCY_WARMUP = 20
LATENCY_RUNS   = 200

CLASS_NAMES: Dict[int, str] = {
    0: "Speed limit (20km/h)",       1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",       3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",       5: "Speed limit (80km/h)",
    6: "End of speed limit (80)",    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",      9: "No passing",
    10: "No passing (>3.5t)",        11: "Right-of-way",
    12: "Priority road",             13: "Yield",
    14: "Stop",                      15: "No vehicles",
    16: "Vehicles >3.5t prohib.",    17: "No entry",
    18: "General caution",           19: "Dangerous curve left",
    20: "Dangerous curve right",     21: "Double curve",
    22: "Bumpy road",                23: "Slippery road",
    24: "Road narrows right",        25: "Road work",
    26: "Traffic signals",           27: "Pedestrians",
    28: "Children crossing",         29: "Bicycles crossing",
    30: "Beware ice/snow",           31: "Wild animals",
    32: "End of all limits",         33: "Turn right ahead",
    34: "Turn left ahead",           35: "Ahead only",
    36: "Go straight or right",      37: "Go straight or left",
    38: "Keep right",                39: "Keep left",
    40: "Roundabout mandatory",      41: "End of no passing",
    42: "End of no passing (>3.5t)",
}

# Short names for confusion matrix axes (space-constrained)
SHORT_NAMES: Dict[int, str] = {
    0: "20",    1: "30",    2: "50",    3: "60",    4: "70",
    5: "80",    6: "80end", 7: "100",   8: "120",   9: "NoPas",
    10: "NP3t", 11: "ROW",  12: "Prio", 13: "Yld",  14: "Stop",
    15: "NoVeh",16: "V3t",  17: "NoEnt",18: "Caut", 19: "CrvL",
    20: "CrvR", 21: "2Crv", 22: "Bump", 23: "Slip", 24: "Narw",
    25: "Work", 26: "Sig",  27: "Ped",  28: "Chld", 29: "Bike",
    30: "Ice",  31: "Wild", 32: "End",  33: "TrnR", 34: "TrnL",
    35: "Ahd",  36: "SR",   37: "SL",   38: "KpR",  39: "KpL",
    40: "Rdbt", 41: "ENP",  42: "ENP3",
}


# ── Model loaders ──────────────────────────────────────────────────────────────

def load_vit(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Reconstruct ViT architecture from config (no weight download) then
    load our fine-tuned weights from checkpoint.
    """
    logger.info("Loading ViT from %s ...", checkpoint_path)
    config = ViTConfig.from_pretrained(
        "google/vit-base-patch16-224",
        num_labels=NUM_CLASSES,
    )
    model = ViTForImageClassification(config)
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("ViT loaded — val_acc at save: %.4f (epoch %d)",
                ckpt["val_acc"], ckpt["epoch"])
    return model


def load_resnet(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Reconstruct ResNet50 and load fine-tuned weights."""
    logger.info("Loading ResNet50 from %s ...", checkpoint_path)
    model    = resnet50(weights=None)           # no ImageNet weights needed
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    ckpt     = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    logger.info("ResNet50 loaded — val_acc at save: %.4f (epoch %d)",
                ckpt["val_acc"], ckpt["epoch"])
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_vit: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run full test set inference. Returns (all_preds, all_labels).
    is_vit=True uses outputs.logits; is_vit=False uses outputs directly.
    """
    all_preds:  List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in tqdm(loader, desc="Inference", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            out = model(pixel_values=images) if is_vit else model(images)
        logits = out.logits if is_vit else out
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)

    return torch.cat(all_preds), torch.cat(all_labels)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[float, Dict[int, float], np.ndarray]:
    """Returns (overall_acc, per_class_acc, confusion_matrix)."""
    overall_acc = (preds == labels).float().mean().item()

    per_class: Dict[int, float] = {}
    for c in range(NUM_CLASSES):
        mask = labels == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == c).float().mean().item()

    # Confusion matrix
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int32)
    for p, l in zip(preds.numpy(), labels.numpy()):
        cm[l, p] += 1

    return overall_acc, per_class, cm


# ── Latency benchmark ──────────────────────────────────────────────────────────

def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    is_vit: bool,
    input_size: Tuple[int, ...] = (1, 3, 224, 224),
) -> Tuple[float, float]:
    """
    Measure single-image GPU latency using CUDA events (most accurate method).
    Returns (mean_ms, std_ms).
    """
    dummy = torch.randn(*input_size, device=device)

    # Warm up — lets CUDA compile kernels so timing is fair
    for _ in range(LATENCY_WARMUP):
        with torch.no_grad():
            _ = model(pixel_values=dummy) if is_vit else model(dummy)
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(LATENCY_RUNS):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model(pixel_values=dummy) if is_vit else model(dummy)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return float(np.mean(times)), float(np.std(times))


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Plot 43×43 confusion matrix. Diagonal = correct, off-diagonal = errors."""
    fig, ax = plt.subplots(figsize=(18, 16))

    # Normalise rows to [0,1] so colour shows rate not count
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # avoid div by zero
    cm_norm = cm_norm / row_sums

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Recall rate")

    ticks = list(range(NUM_CLASSES))
    tick_labels = [SHORT_NAMES[i] for i in ticks]
    ax.set_xticks(ticks); ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks); ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class",      fontsize=11)
    ax.set_title(title,              fontsize=13, pad=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_per_class_accuracy(
    vit_pc:    Dict[int, float],
    resnet_pc: Dict[int, float],
    save_path: Path,
) -> None:
    """Side-by-side horizontal bar chart, sorted by ViT accuracy ascending."""
    sorted_ids = sorted(vit_pc.keys(), key=lambda c: vit_pc[c])
    names      = [CLASS_NAMES[c] for c in sorted_ids]
    vit_vals   = [vit_pc[c]    * 100 for c in sorted_ids]
    resnet_vals= [resnet_pc.get(c, 0) * 100 for c in sorted_ids]

    y = np.arange(len(sorted_ids))
    h = 0.35

    fig, ax = plt.subplots(figsize=(13, 16))
    ax.barh(y + h/2, vit_vals,    h, label="ViT-base",  color="#3b82f6", alpha=0.85)
    ax.barh(y - h/2, resnet_vals, h, label="ResNet50", color="#f97316", alpha=0.85)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-class Test Accuracy — ViT-base vs ResNet50", fontsize=13)
    ax.set_xlim(80, 101)
    ax.axvline(100, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ── Report ─────────────────────────────────────────────────────────────────────

def print_comparison_table(
    vit_acc:    float,
    resnet_acc: float,
    vit_ms:     float,
    vit_std:    float,
    resnet_ms:  float,
    resnet_std: float,
    vit_pc:     Dict[int, float],
    resnet_pc:  Dict[int, float],
    n_test:     int,
) -> None:
    vit_fps    = 1000 / vit_ms
    resnet_fps = 1000 / resnet_ms

    print(f"\n{'='*65}")
    print(f"  Test Set Evaluation  ({n_test:,} images, never seen during training)")
    print(f"{'='*65}")
    print(f"  {'Model':<18} {'Test Acc':>9}  {'Params':>7}  {'ms/img':>8}  {'FPS':>6}")
    print(f"  {'─'*18} {'─'*9}  {'─'*7}  {'─'*8}  {'─'*6}")
    print(f"  {'ViT-base':<18} {vit_acc*100:>8.3f}%  {'86M':>7}  "
          f"{vit_ms:>6.2f}ms  {vit_fps:>6.0f}")
    print(f"  {'ResNet50':<18} {resnet_acc*100:>8.3f}%  {'23M':>7}  "
          f"{resnet_ms:>6.2f}ms  {resnet_fps:>6.0f}")
    print(f"{'─'*65}")
    print(f"  Gap (ViT - ResNet)   {(vit_acc - resnet_acc)*100:>+8.3f}%")
    faster_model = "ViT" if vit_ms < resnet_ms else "ResNet"
    ratio = max(vit_ms, resnet_ms) / min(vit_ms, resnet_ms)
    print(f"  Speed ratio          {faster_model} is {ratio:.1f}x faster per image (batch=1, GPU)")
    print(f"{'='*65}")

    # Worst 10 classes — both models
    all_classes = sorted(vit_pc.keys(), key=lambda c: vit_pc[c])
    print(f"\n  10 hardest classes for ViT (test set):")
    print(f"  {'Class':<38} {'ViT':>7}  {'ResNet':>7}  {'Winner'}")
    print(f"  {'─'*38} {'─'*7}  {'─'*7}  {'─'*6}")
    for c in all_classes[:10]:
        v = vit_pc[c] * 100
        r = resnet_pc.get(c, 0) * 100
        winner = "ViT" if v > r else ("ResNet" if r > v else "Tie")
        print(f"  [{c:02d}] {CLASS_NAMES[c]:<35} {v:>6.1f}%  {r:>6.1f}%  {winner}")
    print()


def save_csv(
    vit_acc: float, resnet_acc: float,
    vit_ms: float,  resnet_ms: float,
    vit_pc: Dict[int, float], resnet_pc: Dict[int, float],
    save_path: Path,
) -> None:
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name",
                         "vit_test_acc", "resnet_test_acc", "winner"])
        for c in range(NUM_CLASSES):
            v = vit_pc.get(c, 0)
            r = resnet_pc.get(c, 0)
            winner = "vit" if v > r else ("resnet" if r > v else "tie")
            writer.writerow([c, CLASS_NAMES[c], f"{v:.4f}", f"{r:.4f}", winner])
        writer.writerow([])
        writer.writerow(["overall", "ALL", f"{vit_acc:.4f}", f"{resnet_acc:.4f}",
                         "vit" if vit_acc > resnet_acc else "resnet"])
        writer.writerow(["latency_ms", "", f"{vit_ms:.2f}", f"{resnet_ms:.2f}", ""])
    logger.info("Saved: %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Load models ───────────────────────────────────────────────────────────
    vit    = load_vit(CHECKPOINT_DIR / "vit_best.pt",    device)
    resnet = load_resnet(CHECKPOINT_DIR / "resnet_best.pt", device)

    # ── Test dataloaders (separate normalization per model) ───────────────────
    vit_test_ds = GTSRBDataset(
        "test", transform=get_eval_transform(mean=VIT_MEAN, std=VIT_STD)
    )
    resnet_test_ds = GTSRBDataset(
        "test", transform=get_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    )

    vit_loader = DataLoader(
        vit_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    resnet_loader = DataLoader(
        resnet_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    logger.info("Test set: %d images", len(vit_test_ds))

    # ── Inference ─────────────────────────────────────────────────────────────
    logger.info("Running ViT inference ...")
    vit_preds, vit_labels = run_inference(vit, vit_loader, device, is_vit=True)

    logger.info("Running ResNet50 inference ...")
    resnet_preds, _ = run_inference(resnet, resnet_loader, device, is_vit=False)
    resnet_labels   = vit_labels      # same test set, same order

    # ── Metrics ───────────────────────────────────────────────────────────────
    vit_acc,    vit_pc,    vit_cm    = compute_metrics(vit_preds,    vit_labels)
    resnet_acc, resnet_pc, resnet_cm = compute_metrics(resnet_preds, resnet_labels)

    logger.info("ViT    test acc: %.4f", vit_acc)
    logger.info("ResNet test acc: %.4f", resnet_acc)

    # ── Latency benchmark (GPU, batch=1) ─────────────────────────────────────
    logger.info("Benchmarking latency (%d runs, batch=1) ...", LATENCY_RUNS)
    vit_ms,    vit_std    = benchmark_latency(vit,    device, is_vit=True)
    resnet_ms, resnet_std = benchmark_latency(resnet, device, is_vit=False)
    logger.info("ViT    latency: %.2f ± %.2f ms", vit_ms, vit_std)
    logger.info("ResNet latency: %.2f ± %.2f ms", resnet_ms, resnet_std)

    # ── Plots ─────────────────────────────────────────────────────────────────
    logger.info("Generating plots ...")
    plot_confusion_matrix(
        vit_cm,
        title="Confusion Matrix — ViT-base (test set)",
        save_path=OUTPUT_DIR / "confusion_matrix_vit.png",
    )
    plot_confusion_matrix(
        resnet_cm,
        title="Confusion Matrix — ResNet50 (test set)",
        save_path=OUTPUT_DIR / "confusion_matrix_resnet.png",
    )
    plot_per_class_accuracy(
        vit_pc, resnet_pc,
        save_path=OUTPUT_DIR / "per_class_accuracy.png",
    )

    # ── Report ────────────────────────────────────────────────────────────────
    print_comparison_table(
        vit_acc, resnet_acc,
        vit_ms, vit_std,
        resnet_ms, resnet_std,
        vit_pc, resnet_pc,
        n_test=len(vit_test_ds),
    )

    save_csv(
        vit_acc, resnet_acc,
        vit_ms, resnet_ms,
        vit_pc, resnet_pc,
        save_path=OUTPUT_DIR / "test_results.csv",
    )

    logger.info("Evaluation complete. Outputs in %s", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
