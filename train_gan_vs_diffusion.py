#!/usr/bin/env python3
"""
Train a binary image classifier to separate GAN-generated vs diffusion-generated images.

Datasets:
  - GAN:        kmk29774/my-gan-training-images
  - Diffusion:  lesc-unifi/dragon (config: ExtraLarge)

Note: authenticate first if datasets are gated:
  huggingface-cli login
"""

import argparse
import csv
import io
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset as TorchDataset, Subset
from torchvision import models, transforms


LABEL_TO_ID = {"gan": 0, "diffusion": 1}
ID_TO_LABEL = {0: "gan", 1: "diffusion"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_image_column(ds: Dataset) -> str:
    preferred = ("image", "img", "images", "pixel_values")
    for name in preferred:
        if name in ds.column_names:
            return name

    for name, feat in ds.features.items():
        if feat.__class__.__name__.lower() == "image":
            return name

    first = ds[0]
    for name, value in first.items():
        if isinstance(value, Image.Image):
            return name
        if isinstance(value, dict) and any(k in value for k in ("bytes", "path", "array")):
            return name

    raise ValueError(
        f"Could not infer image column. Available columns: {ds.column_names}. "
        "Use a dataset with an image-like column."
    )


def decode_image(value) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
        if value.get("array") is not None:
            return Image.fromarray(np.array(value["array"])).convert("RGB")

    raise TypeError(f"Unsupported image value type: {type(value)}")


def flatten_splits(ds_or_dict) -> Dataset:
    if isinstance(ds_or_dict, Dataset):
        return ds_or_dict
    if isinstance(ds_or_dict, DatasetDict):
        splits = [ds_or_dict[k] for k in ds_or_dict.keys()]
        if not splits:
            raise ValueError("DatasetDict has no splits.")
        if len(splits) == 1:
            return splits[0]
        return concatenate_datasets(splits)
    raise TypeError(f"Unexpected dataset type: {type(ds_or_dict)}")


def patch_hf_symlink_fallback() -> None:
    """Fallback to file copy on Windows when symlink privileges are unavailable."""
    try:
        from huggingface_hub import file_download
    except Exception:
        return

    original = file_download._create_symlink

    def _safe_create_symlink(src_rel_or_abs: str, abs_dst: str, new_blob: bool = False):
        try:
            return original(src_rel_or_abs, abs_dst, new_blob=new_blob)
        except OSError as exc:
            if os.name != "nt" or getattr(exc, "winerror", None) != 1314:
                raise

            abs_dst_path = Path(abs_dst)
            abs_dst_path.parent.mkdir(parents=True, exist_ok=True)

            src_path = Path(src_rel_or_abs)
            if not src_path.is_absolute():
                src_path = (abs_dst_path.parent / src_path).resolve()

            if abs_dst_path.exists() or abs_dst_path.is_symlink():
                abs_dst_path.unlink()

            shutil.copy2(str(src_path), str(abs_dst_path))

    file_download._create_symlink = _safe_create_symlink


def load_dataset_safe(
    name: str,
    config: Optional[str],
    token: Optional[str],
    split: Optional[str] = None,
    data_files=None,
):
    try:
        return load_dataset(name, config, token=token, split=split, data_files=data_files)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            patch_hf_symlink_fallback()
            return load_dataset(name, config, token=token, split=split, data_files=data_files)
        raise


class HFClassDataset(TorchDataset):
    def __init__(self, hf_dataset: Dataset, image_column: str, label_id: int, transform=None):
        self.ds = hf_dataset
        self.image_column = image_column
        self.label_id = label_id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        image = decode_image(row[self.image_column])
        if self.transform is not None:
            image = self.transform(image)
        return image, self.label_id


def split_indices(
    n: int,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    max_samples: Optional[int],
) -> Tuple[Sequence[int], Sequence[int], Sequence[int]]:
    indices = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    if max_samples is not None and max_samples > 0:
        indices = indices[: min(max_samples, len(indices))]

    n_total = len(indices)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val - n_test

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            "Invalid split sizes. Adjust val_ratio/test_ratio/max_samples so each split has at least one sample."
        )

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]
    return train_idx, val_idx, test_idx


def build_model(backbone: str, pretrained: bool) -> nn.Module:
    backbone = backbone.lower()

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    if backbone == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, 2)
        return model

    raise ValueError("Unsupported backbone. Use one of: resnet18, resnet50")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

    metrics = {
        "loss": total_loss / max(1, len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = 0.5

    report = classification_report(
        y_true,
        y_pred,
        target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
        output_dict=True,
        zero_division=0,
    )
    metrics["precision_gan"] = report["gan"]["precision"]
    metrics["recall_gan"] = report["gan"]["recall"]
    metrics["precision_diffusion"] = report["diffusion"]["precision"]
    metrics["recall_diffusion"] = report["diffusion"]["recall"]

    return metrics


def write_history(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_val_f1: float,
    history: List[Dict[str, float]],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_val_f1": best_val_f1,
        "history": history,
    }
    torch.save(ckpt, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GAN-vs-diffusion image classifier")
    parser.add_argument("--gan_dataset", type=str, default="kmk29774/my-gan-training-images")
    parser.add_argument("--gan_config", type=str, default=None)
    parser.add_argument("--diffusion_dataset", type=str, default="lesc-unifi/dragon")
    parser.add_argument("--diffusion_config", type=str, default="ExtraLarge")
    parser.add_argument("--hf_token", type=str, default=None, help="Optional HF token; otherwise uses HF_TOKEN env var")
    parser.add_argument("--output_dir", type=str, default="models_gan_vs_diffusion")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples_per_class", type=int, default=1000)
    parser.add_argument(
        "--diffusion_tar_limit",
        type=int,
        default=0,
        help="Limit diffusion dataset to first N train tar shards (0 = no tar limit).",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output_dir")
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0 or (args.val_ratio + args.test_ratio) >= 1:
        raise ValueError("Use val/test ratios in (0,1) and ensure val_ratio + test_ratio < 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    token = args.hf_token or os.getenv("HF_TOKEN")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading GAN dataset...")
    gan_raw = load_dataset_safe(args.gan_dataset, args.gan_config, token=token)
    gan_ds = flatten_splits(gan_raw)
    gan_img_col = pick_image_column(gan_ds)
    print(f"GAN samples: {len(gan_ds)} | image column: {gan_img_col}")

    print("Loading diffusion dataset...")
    diffusion_cap = 9984
    max_per_class = args.max_samples_per_class if args.max_samples_per_class > 0 else None
    diffusion_requested = diffusion_cap if max_per_class is None else min(diffusion_cap, max_per_class)
    diffusion_split = f"train[:{diffusion_requested}]"
    diffusion_data_files = None

    if args.diffusion_tar_limit > 0:
        # Dragon shards follow train/dragon_train_000.tar naming; limit to first N shards.
        tar_files = [f"train/dragon_train_{i:03d}.tar" for i in range(args.diffusion_tar_limit)]
        diffusion_data_files = {"train": tar_files}
        diffusion_split = "train"
        print(f"Applying diffusion tar limit: {args.diffusion_tar_limit} shard(s)")

    try:
        diff_raw = load_dataset_safe(
            args.diffusion_dataset,
            args.diffusion_config,
            token=token,
            split=diffusion_split,
            data_files=diffusion_data_files,
        )
        print(f"Loaded diffusion split: {diffusion_split}")
    except Exception:
        # Fallback if dataset does not support slicing syntax.
        diff_raw = load_dataset_safe(
            args.diffusion_dataset,
            args.diffusion_config,
            token=token,
            data_files=diffusion_data_files,
        )

    diff_ds = flatten_splits(diff_raw)
    diff_img_col = pick_image_column(diff_ds)
    print(f"Diffusion samples: {len(diff_ds)} | image column: {diff_img_col}")

    # Always cap diffusion samples to keep class balance against GAN dataset size.
    diffusion_max_samples = min(len(diff_ds), diffusion_cap)
    if max_per_class is not None:
        diffusion_max_samples = min(diffusion_max_samples, max_per_class)

    print(f"Applying diffusion sample cap: {diffusion_max_samples}")
    gan_train_idx, gan_val_idx, gan_test_idx = split_indices(
        n=len(gan_ds),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        max_samples=max_per_class,
    )
    diff_train_idx, diff_val_idx, diff_test_idx = split_indices(
        n=len(diff_ds),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed + 1,
        max_samples=diffusion_max_samples,
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    gan_train = Subset(HFClassDataset(gan_ds, gan_img_col, LABEL_TO_ID["gan"], train_tf), gan_train_idx)
    gan_val = Subset(HFClassDataset(gan_ds, gan_img_col, LABEL_TO_ID["gan"], eval_tf), gan_val_idx)
    gan_test = Subset(HFClassDataset(gan_ds, gan_img_col, LABEL_TO_ID["gan"], eval_tf), gan_test_idx)

    diff_train = Subset(HFClassDataset(diff_ds, diff_img_col, LABEL_TO_ID["diffusion"], train_tf), diff_train_idx)
    diff_val = Subset(HFClassDataset(diff_ds, diff_img_col, LABEL_TO_ID["diffusion"], eval_tf), diff_val_idx)
    diff_test = Subset(HFClassDataset(diff_ds, diff_img_col, LABEL_TO_ID["diffusion"], eval_tf), diff_test_idx)

    train_set = ConcatDataset([gan_train, diff_train])
    val_set = ConcatDataset([gan_val, diff_val])
    test_set = ConcatDataset([gan_test, diff_test])

    print(f"Train size: {len(train_set)}")
    print(f"Val size:   {len(val_set)}")
    print(f"Test size:  {len(test_set)}")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(args.backbone, pretrained=not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history: List[Dict[str, float]] = []
    best_val_f1 = -1.0
    best_path = out_dir / "best_model_weights.pt"
    checkpoint_path = out_dir / "last_checkpoint.pt"
    start_epoch = 1

    if args.resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        best_val_f1 = float(checkpoint.get("best_val_f1", -1.0))
        history = list(checkpoint.get("history", []))
        start_epoch = int(checkpoint["epoch"]) + 1
        print(f"Resumed at epoch {start_epoch} with best_val_f1={best_val_f1:.4f}")
    elif args.resume:
        print("Resume requested but no checkpoint found. Starting fresh training.")

    if start_epoch > args.epochs:
        print(
            f"Checkpoint epoch ({start_epoch - 1}) is already >= requested epochs ({args.epochs}). "
            "Skipping train loop and evaluating best model."
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        train_loss = running_loss / max(1, total)
        train_acc = correct / max(1, total)
        val_metrics = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "val_auc": val_metrics["auc"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} val_auc={val_metrics['auc']:.4f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)

        # Save continuation state each epoch so training can resume safely after interruptions.
        save_checkpoint(
            checkpoint_path,
            epoch,
            model,
            optimizer,
            scheduler,
            scaler,
            best_val_f1,
            history,
        )
        write_history(out_dir / "training_history.csv", history)

    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    print("\nFinal test metrics:")
    print(json.dumps(test_metrics, indent=2))

    config = {
        "task": "gan_vs_diffusion",
        "gan_dataset": args.gan_dataset,
        "gan_config": args.gan_config,
        "diffusion_dataset": args.diffusion_dataset,
        "diffusion_config": args.diffusion_config,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "backbone": args.backbone,
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "max_samples_per_class": max_per_class,
        "seed": args.seed,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
        "weights_path": str(best_path),
    }

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    write_history(out_dir / "training_history.csv", history)
    print(f"\nSaved artifacts to: {out_dir}")
    print(f"- Weights: {best_path}")
    print(f"- Config:  {out_dir / 'config.json'}")
    print(f"- History: {out_dir / 'training_history.csv'}")
    print(f"- Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
