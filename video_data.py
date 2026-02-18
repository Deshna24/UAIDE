from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _list_videos(folder: Path) -> List[Path]:
    files = []
    for ext in VIDEO_EXTENSIONS:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)


def _split_list(items: List[Path], val_split: float, seed: int) -> Tuple[List[Path], List[Path]]:
    if val_split <= 0.0:
        return items, []
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    split = int(round(len(items) * (1.0 - val_split)))
    train_idx = idx[:split]
    val_idx = idx[split:]
    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    return train_items, val_items


class FaceCropper:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 4, min_size: int = 30, margin: float = 0.25):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size
        self.margin = margin
        self.cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def crop(self, frame_rgb: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        if len(faces) == 0:
            return frame_rgb
        # Choose the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        # Add margin
        mx = int(w * self.margin)
        my = int(h * self.margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(frame_rgb.shape[1], x + w + mx)
        y1 = min(frame_rgb.shape[0], y + h + my)
        return frame_rgb[y0:y1, x0:x1]


def read_video_frames(
    video_path: Path,
    frames_per_video: int,
    frame_stride: int,
    face_cropper: FaceCropper = None,
) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if face_cropper is not None:
                frame_rgb = face_cropper.crop(frame_rgb)
            frames.append(frame_rgb)
            if len(frames) >= frames_per_video:
                break
        idx += 1
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")

    # Pad with last frame if needed
    while len(frames) < frames_per_video:
        frames.append(frames[-1])

    return frames


class VideoDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
        max_videos_per_class: int = None,
        frames_per_video: int = 16,
        frame_stride: int = 4,
        face_detection: bool = True,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        real_dir = self.root_dir / "videos_real"
        fake_dir = self.root_dir / "videos_fake"
        if not real_dir.exists() or not fake_dir.exists():
            raise FileNotFoundError("Expected videos_real and videos_fake under root_dir")

        real_videos = _list_videos(real_dir)
        fake_videos = _list_videos(fake_dir)
        if max_videos_per_class:
            real_videos = real_videos[:max_videos_per_class]
            fake_videos = fake_videos[:max_videos_per_class]

        real_train, real_val = _split_list(real_videos, val_split, seed)
        fake_train, fake_val = _split_list(fake_videos, val_split, seed)

        if split == "train":
            self.samples = [(p, 0) for p in real_train] + [(p, 1) for p in fake_train]
        else:
            self.samples = [(p, 0) for p in real_val] + [(p, 1) for p in fake_val]

        self.frames_per_video = frames_per_video
        self.frame_stride = frame_stride
        self.transform = transform
        self.face_cropper = FaceCropper() if face_detection else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        video_path, label = self.samples[idx]
        frames = read_video_frames(
            video_path,
            frames_per_video=self.frames_per_video,
            frame_stride=self.frame_stride,
            face_cropper=self.face_cropper,
        )
        tensors = []
        for frame_rgb in frames:
            img = Image.fromarray(frame_rgb)
            if self.transform is not None:
                img = self.transform(img)
            tensors.append(img)
        video_tensor = torch.stack(tensors, dim=0)  # (T, C, H, W)
        return video_tensor, torch.tensor(label, dtype=torch.long)
