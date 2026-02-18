import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from video_data import VideoDataset
from video_model import ResNetLSTM


def build_transforms():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def accuracy_from_logits(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, frame_loss_weight):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for videos, labels in loader:
        videos = videos.to(device)
        labels = labels.to(device)

        frame_logits, video_logits = model(videos)
        loss_video = criterion(video_logits, labels)

        b, t, c = frame_logits.shape
        frame_targets = labels.repeat_interleave(t)
        loss_frame = criterion(frame_logits.view(b * t, c), frame_targets)

        loss = loss_video + frame_loss_weight * loss_frame

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(video_logits, labels)

    return total_loss / max(1, len(loader)), total_acc / max(1, len(loader))


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)

            _, video_logits = model(videos)
            loss = criterion(video_logits, labels)

            total_loss += loss.item()
            total_acc += accuracy_from_logits(video_logits, labels)

    return total_loss / max(1, len(loader)), total_acc / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="SDFVD/SDFVD")
    parser.add_argument("--out", type=str, default="video_resnet_lstm.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--override_config", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frames_per_video", type=int, default=16)
    parser.add_argument("--frame_stride", type=int, default=4)
    parser.add_argument("--max_videos_per_class", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_face_detection", action="store_true")
    parser.add_argument("--frame_loss_weight", type=float, default=0.3)
    parser.add_argument("--temporal_pool", type=str, choices=["mean", "last"], default="mean")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--no_bidirectional", action="store_false", dest="bidirectional")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    resume_path = Path(args.resume) if args.resume else None
    checkpoint = None
    if resume_path and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location="cpu")
        if not args.override_config and isinstance(checkpoint, dict) and "config" in checkpoint:
            cfg = checkpoint["config"]
            args.hidden_size = cfg.get("hidden_size", args.hidden_size)
            args.num_layers = cfg.get("num_layers", args.num_layers)
            args.bidirectional = cfg.get("bidirectional", args.bidirectional)
            args.temporal_pool = cfg.get("temporal_pool", args.temporal_pool)
            args.no_pretrained = not cfg.get("pretrained", not args.no_pretrained)
            args.frames_per_video = cfg.get("frames_per_video", args.frames_per_video)
            args.frame_stride = cfg.get("frame_stride", args.frame_stride)
            args.no_face_detection = not cfg.get("face_detection", not args.no_face_detection)

    transform = build_transforms()
    train_ds = VideoDataset(
        root_dir=args.dataset,
        split="train",
        val_split=args.val_split,
        seed=args.seed,
        max_videos_per_class=args.max_videos_per_class,
        frames_per_video=args.frames_per_video,
        frame_stride=args.frame_stride,
        face_detection=not args.no_face_detection,
        transform=transform,
    )
    val_ds = VideoDataset(
        root_dir=args.dataset,
        split="val",
        val_split=args.val_split,
        seed=args.seed,
        max_videos_per_class=args.max_videos_per_class,
        frames_per_video=args.frames_per_video,
        frame_stride=args.frame_stride,
        face_detection=not args.no_face_detection,
        transform=transform,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = ResNetLSTM(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional,
        temporal_pool=args.temporal_pool,
        pretrained=not args.no_pretrained,
    )
    model.to(device)

    if checkpoint and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"], strict=True)
        print(f"Resumed model weights from {resume_path}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    start_epoch = 0
    if checkpoint:
        best_val_acc = float(checkpoint.get("best_val_acc", best_val_acc))
        start_epoch = int(checkpoint.get("epoch", start_epoch))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_epochs = start_epoch + args.epochs
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.frame_loss_weight
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {start_epoch + epoch + 1}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": start_epoch + epoch + 1,
                    "best_val_acc": best_val_acc,
                    "config": {
                        "hidden_size": args.hidden_size,
                        "num_layers": args.num_layers,
                        "bidirectional": args.bidirectional,
                        "temporal_pool": args.temporal_pool,
                        "pretrained": not args.no_pretrained,
                        "frames_per_video": args.frames_per_video,
                        "frame_stride": args.frame_stride,
                        "face_detection": not args.no_face_detection,
                    },
                },
                out_path,
            )
            print(f"Saved best model to {out_path}")


if __name__ == "__main__":
    main()
