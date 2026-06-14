"""
Block 2 — ViT Fine-tuning Script

Fine-tunes google/vit-base-patch16-224 on GTSRB (43 classes).

Run from project root:
    ../.venv/Scripts/python training/train_vit.py

Key decisions:
  - Weighted CrossEntropyLoss for class imbalance (Option A)
  - Separate LR for backbone vs head (head is randomly initialised, needs higher LR)
  - AdamW + cosine schedule with linear warmup (standard for ViT fine-tuning)
  - fp16 mixed precision via torch.amp (fits 6GB VRAM at batch=32)
  - Metrics logged to console (tqdm) + CSV file after every epoch
  - Best checkpoint saved on val accuracy improvement
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
from tqdm import tqdm
from transformers import ViTForImageClassification

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import GTSRBDataset, NUM_CLASSES
from data.transforms import get_eval_transform, get_train_transform

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path(__file__).parent / "logs" / "train_vit.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    """All training hyperparameters in one place. Change here, nowhere else."""

    model_name: str = "google/vit-base-patch16-224"
    num_classes: int = NUM_CLASSES

    # Data
    batch_size: int = 32          # max safe for 6GB VRAM with ViT-base + fp16
    num_workers: int = 4
    val_fraction: float = 0.15
    seed: int = 42

    # Optimiser
    # Two LRs: backbone has pretrained weights (low LR preserves them),
    # head is randomly initialised (high LR lets it learn fast).
    lr_backbone: float = 2e-5
    lr_head: float = 1e-4
    weight_decay: float = 0.01

    # Schedule
    num_epochs: int = 20
    warmup_epochs: int = 2        # linear warmup before cosine decay kicks in

    # Hardware
    use_amp: bool = True          # fp16 mixed precision — essential for 6GB VRAM

    # Paths
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    log_csv: Path = field(default_factory=lambda: Path("training/logs/vit_metrics.csv"))


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Weighted loss ──────────────────────────────────────────────────────────────
def build_class_weights(counts: Dict[int, int], num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Rare classes get higher weight so the model is penalised more
    for misclassifying them. Weights normalised so mean weight = 1.0
    (preserves the scale of the loss — makes LR tuning more predictable).
    """
    total = sum(counts.values())
    weights = torch.tensor(
        [total / counts.get(i, 1) for i in range(num_classes)],
        dtype=torch.float32,
    )
    weights = weights / weights.mean()  # normalise: mean weight = 1.0
    return weights


# ── Metrics ────────────────────────────────────────────────────────────────────
def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    return (logits.argmax(dim=1) == labels).float().mean().item()


def per_class_accuracy(
    all_preds: torch.Tensor,
    all_labels: torch.Tensor,
    num_classes: int,
) -> Dict[int, float]:
    """Per-class accuracy over the full validation set."""
    result: Dict[int, float] = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            result[c] = (all_preds[mask] == c).float().mean().item()
    return result


def print_worst_classes(
    pc_acc: Dict[int, float],
    class_names: Dict[int, str],
    n: int = 5,
) -> None:
    """Print the N worst-performing classes on validation set."""
    sorted_classes = sorted(pc_acc.items(), key=lambda x: x[1])
    print(f"\n  {n} worst val classes:")
    for class_id, acc in sorted_classes[:n]:
        name = class_names.get(class_id, f"class_{class_id}")
        bar = "-" * int(acc * 20)
        print(f"    [{class_id:02d}] {name:<38} {acc*100:5.1f}%  |{bar}")


# ── One training epoch ─────────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    use_amp: bool,
) -> Tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, avg_accuracy)."""
    model.train()
    total_loss, total_acc, steps = 0.0, 0.0, 0

    pbar = tqdm(loader, desc=f"Train {epoch+1:02d}/{num_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        batch_acc = accuracy(outputs.logits.detach(), labels)
        total_loss += loss.item()
        total_acc  += batch_acc
        steps      += 1

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc":  f"{batch_acc:.3f}",
            "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
        })

    return total_loss / steps, total_acc / steps


# ── One validation epoch ───────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, float, Dict[int, float]]:
    """Run validation. Returns (avg_loss, avg_accuracy, per_class_acc)."""
    model.eval()
    total_loss, steps = 0.0, 0
    all_preds:  List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    pbar = tqdm(loader, desc="Val  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(pixel_values=images)
            loss = criterion(outputs.logits, labels)

        total_loss  += loss.item()
        steps       += 1
        all_preds.append(outputs.logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())

    all_preds_t  = torch.cat(all_preds)
    all_labels_t = torch.cat(all_labels)
    avg_acc      = (all_preds_t == all_labels_t).float().mean().item()
    pc_acc       = per_class_accuracy(all_preds_t, all_labels_t, NUM_CLASSES)

    return total_loss / steps, avg_acc, pc_acc


# ── CSV logger ─────────────────────────────────────────────────────────────────
def init_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc",
            "lr_backbone", "lr_head", "epoch_time_s",
        ])


def log_csv(path: Path, row: List) -> None:
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow(row)


# ── Epoch summary ──────────────────────────────────────────────────────────────
def print_epoch_summary(
    epoch: int,
    num_epochs: int,
    train_loss: float,
    train_acc: float,
    val_loss: float,
    val_acc: float,
    best_val_acc: float,
    best_epoch: int,
    lr_backbone: float,
    lr_head: float,
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

    # ── Datasets & loaders ────────────────────────────────────────────────────
    logger.info("Building datasets ...")
    train_transform = get_train_transform()
    eval_transform  = get_eval_transform()

    train_ds = GTSRBDataset("train", transform=train_transform,
                            val_fraction=config.val_fraction, seed=config.seed)
    val_ds   = GTSRBDataset("val",   transform=eval_transform,
                            val_fraction=config.val_fraction, seed=config.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,   # no grad → can fit 2× batch
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # ── Weighted loss ─────────────────────────────────────────────────────────
    counts  = train_ds.class_counts()
    weights = build_class_weights(counts, config.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    logger.info("Weighted CrossEntropyLoss — min weight: %.3f  max weight: %.3f",
                weights.min().item(), weights.max().item())

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Loading %s ...", config.model_name)
    model = ViTForImageClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_classes,
        ignore_mismatched_sizes=True,   # replaces 1000-class head with 43-class head
    )
    model = model.to(device)
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters — total: %dM  trainable: %dM",
                total_params // 1_000_000, trainable_params // 1_000_000)

    # ── Optimiser (separate LR for backbone and head) ─────────────────────────
    optimizer = torch.optim.AdamW(
        [
            {"params": model.vit.parameters(),        "lr": config.lr_backbone},
            {"params": model.classifier.parameters(), "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )

    # ── LR schedule: linear warmup → cosine decay ─────────────────────────────
    total_steps  = len(train_loader) * config.num_epochs
    warmup_steps = len(train_loader) * config.warmup_epochs

    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    # ── AMP scaler ────────────────────────────────────────────────────────────
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # ── Class names (for worst-class display) ─────────────────────────────────
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

    # ── CSV init ──────────────────────────────────────────────────────────────
    init_csv(config.log_csv)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_epoch   = 0

    logger.info("Starting training — %d epochs", config.num_epochs)

    for epoch in range(config.num_epochs):
        t0 = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, scaler, device,
            epoch, config.num_epochs, config.use_amp,
        )

        # Validate
        val_loss, val_acc, pc_acc = val_epoch(
            model, val_loader, criterion, device, config.use_amp
        )

        elapsed = time.time() - t0
        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head     = optimizer.param_groups[1]["lr"]

        # Summary
        print_epoch_summary(
            epoch, config.num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            best_val_acc, best_epoch,
            lr_backbone, lr_head, elapsed,
        )

        # Worst classes — visible every epoch to track imbalance handling
        print_worst_classes(pc_acc, class_names, n=5)

        # CSV log
        log_csv(config.log_csv, [
            epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}",
            f"{val_loss:.6f}",  f"{val_acc:.6f}",
            f"{lr_backbone:.2e}", f"{lr_head:.2e}", f"{elapsed:.1f}",
        ])

        # Checkpoint on improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch + 1
            ckpt_path    = config.checkpoint_dir / "vit_best.pt"
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

    logger.info("Training complete. Best val acc: %.4f (epoch %d)",
                best_val_acc, best_epoch)
    logger.info("Metrics CSV: %s", config.log_csv.resolve())
    logger.info("Best checkpoint: %s", (config.checkpoint_dir / "vit_best.pt").resolve())


if __name__ == "__main__":
    main()
