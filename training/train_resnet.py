"""
Block 3 — ResNet50 Baseline Training Script

Fine-tunes ResNet50 (ImageNet pretrained) on GTSRB (43 classes).
Mirrors train_vit.py in structure — same loop, same metrics, same logging.
Results feed directly into Block 5 comparison table (ViT vs ResNet50).

Key differences from train_vit.py:
  - torchvision ResNet50 instead of HuggingFace ViT
  - Replaces model.fc (2048 → 43) instead of model.classifier (768 → 43)
  - Forward pass returns logits directly (no .logits attribute)
  - ImageNet normalization (mean/std != 0.5) — ResNet50 pretrained with ImageNet stats
  - Higher LRs — CNNs tolerate higher LR than ViT without destroying pretrained weights

Run from project root:
    ../.venv/Scripts/python training/train_resnet.py
"""

import csv
import logging
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GTSRBDataset, NUM_CLASSES
from data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform, get_train_transform

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "logs" / "train_resnet.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """ResNet50 training hyperparameters."""

    num_classes: int = NUM_CLASSES

    # Data
    batch_size: int = 32
    num_workers: int = 4
    val_fraction: float = 0.15
    seed: int = 42

    # Optimiser
    # ResNet tolerates higher LR than ViT — CNN inductive biases make it
    # more stable during fine-tuning. 5x higher than ViT backbone LR.
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4

    # Schedule
    num_epochs: int = 20
    warmup_epochs: int = 2

    # Hardware
    use_amp: bool = True

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_csv: Path = field(default_factory=lambda: Path("training/logs/resnet_metrics.csv"))


# ── Helpers (identical logic to train_vit.py) ──────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised so mean weight = 1.0."""
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32,
    )
    return weights / weights.mean()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def per_class_accuracy(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    result: Dict[int, float] = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            result[c] = (all_preds[mask] == c).float().mean().item()
    return result


def print_worst_classes(pc_acc: Dict[int, float], class_names: Dict[int, str], n: int = 5) -> None:
    sorted_classes = sorted(pc_acc.items(), key=lambda x: x[1])
    print(f"\n  {n} worst val classes:")
    for class_id, acc in sorted_classes[:n]:
        name = class_names.get(class_id, f"class_{class_id}")
        bar = "-" * int(acc * 20)
        print(f"    [{class_id:02d}] {name:<38} {acc*100:5.1f}%  |{bar}")


def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc",
            "lr_backbone", "lr_head", "epoch_time_s",
        ])


def log_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


def print_epoch_summary(
    epoch: int, num_epochs: int,
    train_loss: float, train_acc: float,
    val_loss: float, val_acc: float,
    best_val_acc: float, best_epoch: int,
    lr_backbone: float, lr_head: float,
    elapsed: float,
) -> None:
    print(f"\n{'='*65}")
    print(f"  Epoch {epoch+1:02d}/{num_epochs}   ({elapsed:.0f}s)")
    print(f"{'─'*65}")
    print(f"  {'':10}  {'loss':>10}  {'accuracy':>10}")
    print(f"  {'train':10}  {train_loss:>10.4f}  {train_acc:>10.4f}")
    print(f"  {'val':10}  {val_loss:>10.4f}  {val_acc:>10.4f}")
    print(f"{'─'*65}")
    print(f"  LR backbone : {lr_backbone:.2e}   LR head : {lr_head:.2e}")
    print(f"  Best val acc: {best_val_acc:.4f} (epoch {best_epoch:02d})")
    print(f"{'='*65}\n")


# ── Train / Val loops ──────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    use_amp: bool,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train {epoch+1:02d}/{num_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)          # ResNet returns logits directly
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_acc = accuracy(logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += batch_acc
        steps      += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{batch_acc:.3f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
        })

    return total_loss / steps, total_acc / steps


@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    total_loss, steps = 0.0, 0
    all_preds:  List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        total_loss  += loss.item()
        steps       += 1
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    all_preds_t  = torch.cat(all_preds)
    all_labels_t = torch.cat(all_labels)
    avg_acc      = (all_preds_t == all_labels_t).float().mean().item()
    pc_acc       = per_class_accuracy(all_preds_t, all_labels_t, NUM_CLASSES)

    return total_loss / steps, avg_acc, pc_acc


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    config = TrainConfig()
    set_seed(config.seed)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s  |  VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Datasets — ImageNet normalization for ResNet50 ─────────────────────────
    logger.info("Building datasets ...")
    train_transform = get_train_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    eval_transform  = get_eval_transform(mean=IMAGENET_MEAN,  std=IMAGENET_STD)

    train_ds = GTSRBDataset("train", transform=train_transform,
                            val_fraction=config.val_fraction, seed=config.seed)
    val_ds   = GTSRBDataset("val",   transform=eval_transform,
                            val_fraction=config.val_fraction, seed=config.seed)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=True, persistent_workers=True,
    )
    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ── Weighted loss ──────────────────────────────────────────────────────────
    counts    = train_ds.class_counts()
    weights   = build_class_weights(counts, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info("Weighted CrossEntropyLoss — min weight: %.3f  max weight: %.3f",
                weights.min().item(), weights.max().item())

    # ── Model ──────────────────────────────────────────────────────────────────
    logger.info("Loading ResNet50 (IMAGENET1K_V2 weights) ...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace the final fully-connected layer: 2048 → 43
    # ResNet50 fc: Linear(2048, 1000) → Linear(2048, 43)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters — total: %dM  trainable: %dM",
                total_params // 1_000_000, trainable_params // 1_000_000)

    # ── Optimiser — separate LR for backbone vs head ───────────────────────────
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]
    head_params     = list(model.fc.parameters())

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": config.lr_backbone},
            {"params": head_params,     "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )

    # ── Schedule: linear warmup → cosine decay ─────────────────────────────────
    total_steps  = len(train_loader) * config.num_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ── Class names ────────────────────────────────────────────────────────────
    class_names = {
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

    init_csv(config.log_csv)

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0

    logger.info("Starting training — %d epochs", config.num_epochs)

    for epoch in range(config.num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device,
            epoch, config.num_epochs, config.use_amp,
        )
        val_loss, val_acc, pc_acc = val_epoch(
            model, val_loader, criterion, device, config.use_amp
        )

        elapsed     = time.time() - t0
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head     = optimizer.param_groups[1]["lr"]

        print_epoch_summary(
            epoch, config.num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            best_val_acc, best_epoch,
            lr_backbone, lr_head, elapsed,
        )
        print_worst_classes(pc_acc, class_names, n=5)

        log_csv(config.log_csv, [
            epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}", f"{val_acc:.6f}",
            f"{lr_backbone:.2e}", f"{lr_head:.2e}", f"{elapsed:.1f}",
        ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            ckpt_path    = config.checkpoint_dir / "resnet_best.pt"
            torch.save(
                {
                    "epoch":                epoch + 1,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc":              val_acc,
                    "val_loss":             val_loss,
                    "config":               config,
                    "per_class_acc":        pc_acc,
                },
                ckpt_path,
            )
            logger.info("New best — val_acc: %.4f  saved to %s", val_acc, ckpt_path)

    logger.info("Training complete. Best val acc: %.4f (epoch %d)", best_val_acc, best_epoch)
    logger.info("Metrics CSV: %s", config.log_csv.resolve())
    logger.info("Best checkpoint: %s", (config.checkpoint_dir / "resnet_best.pt").resolve())


if __name__ == "__main__":
    main()
