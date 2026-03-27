from pathlib import Path
from typing import List, Tuple
import io
import tempfile

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


class HuggingFaceVideoDataset(Dataset):
    """Dataset class for HuggingFace deepfake-videos-dataset"""
    def __init__(
        self,
        dataset_name: str = "UniDataPro/deepfake-videos-dataset",
        split: str = "train",
        val_split: float = 0.2,
        seed: int = 42,
        max_videos_per_class: int = None,
        frames_per_video: int = 16,
        frame_stride: int = 4,
        face_detection: bool = True,
        transform=None,
    ):
        """
        Load video dataset from HuggingFace Hub.
        
        Args:
            dataset_name: HuggingFace dataset name
            split: 'train' or 'val' 
            val_split: validation split ratio
            seed: random seed for splitting
            max_videos_per_class: max videos per class (None = all)
            frames_per_video: number of frames to extract per video
            frame_stride: stride for frame extraction
            face_detection: whether to crop faces
            transform: torchvision transforms
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        print(f"Loading HuggingFace dataset: {dataset_name}")
        self.hf_dataset = load_dataset(dataset_name, trust_remote_code=True)
        
        # Extract real and fake samples
        real_samples = []
        fake_samples = []
        
        # Handle different possible dataset structures
        for split_name in self.hf_dataset.column_names if hasattr(self.hf_dataset, 'column_names') else []:
            pass
        
        # Try to load from main dataset
        if isinstance(self.hf_dataset, dict):
            # If it has multiple splits
            main_split = self.hf_dataset.get('train', self.hf_dataset.get('validation', list(self.hf_dataset.values())[0]))
        else:
            main_split = self.hf_dataset
        
        # Identify real/fake based on common column patterns
        label_column = None
        for col in ['label', 'classification', 'deepfake', 'fake', 'is_fake', 'is_deepfake']:
            if col in main_split.column_names:
                label_column = col
                break
        
        # Identify video column
        video_column = None
        for col in ['video', 'video_path', 'video_file', 'video_bytes']:
            if col in main_split.column_names:
                video_column = col
                break
        
        if video_column is None:
            # If no explicit video column, use first column that might contain video data
            video_column = main_split.column_names[0]
        
        print(f"Using video column: {video_column}, label column: {label_column}")
        
        # Load samples
        for idx, sample in enumerate(main_split):
            try:
                video_data = sample[video_column]
                
                # Determine label (assuming binary: 0=real, 1=fake)
                if label_column and label_column in sample:
                    label_val = sample[label_column]
                    # Normalize label to 0 or 1
                    if isinstance(label_val, str):
                        label = 0 if label_val.lower() in ['real', 'authentic', '0'] else 1
                    else:
                        label = int(label_val) if label_val else 0
                else:
                    label = 0  # Default to real if no label
                
                # Store sample info
                sample_info = {
                    'idx': idx,
                    'video_data': video_data,
                    'label': label,
                }
                
                if label == 0:
                    real_samples.append(sample_info)
                else:
                    fake_samples.append(sample_info)
            except Exception as e:
                print(f"Skipping sample {idx}: {e}")
                continue
        
        print(f"Loaded {len(real_samples)} real and {len(fake_samples)} fake videos")
        
        # Limit per class if specified
        if max_videos_per_class:
            real_samples = real_samples[:max_videos_per_class]
            fake_samples = fake_samples[:max_videos_per_class]
        
        # Split into train/val
        rng = np.random.RandomState(seed)
        
        # Shuffle and split real samples
        real_idx = np.arange(len(real_samples))
        rng.shuffle(real_idx)
        real_split = int(round(len(real_samples) * (1.0 - val_split)))
        real_train = [real_samples[i] for i in real_idx[:real_split]]
        real_val = [real_samples[i] for i in real_idx[real_split:]]
        
        # Shuffle and split fake samples
        fake_idx = np.arange(len(fake_samples))
        rng.shuffle(fake_idx)
        fake_split = int(round(len(fake_samples) * (1.0 - val_split)))
        fake_train = [fake_samples[i] for i in fake_idx[:fake_split]]
        fake_val = [fake_samples[i] for i in fake_idx[fake_split:]]
        
        # Select based on split
        if split.lower() == "train":
            self.samples = real_train + fake_train
        else:
            self.samples = real_val + fake_val
        
        self.frames_per_video = frames_per_video
        self.frame_stride = frame_stride
        self.transform = transform
        self.face_cropper = FaceCropper() if face_detection else None
        
        print(f"Created {split} split with {len(self.samples)} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        video_data = sample['video_data']
        label = sample['label']
        
        # Handle different video data formats
        frames = self._extract_frames_from_data(video_data)
        
        if len(frames) < self.frames_per_video:
            # Pad with last frame if needed
            while len(frames) < self.frames_per_video:
                frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Apply frame stride sampling
        frames = frames[::self.frame_stride][:self.frames_per_video]
        
        # Ensure we have exactly frames_per_video frames
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        frames = frames[:self.frames_per_video]
        
        # Apply face cropping if enabled
        if self.face_cropper:
            frames = [self.face_cropper.crop(f) for f in frames]
        
        # Convert to tensors
        tensors = []
        for frame_rgb in frames:
            img = Image.fromarray(frame_rgb)
            if self.transform is not None:
                img = self.transform(img)
            tensors.append(img)
        
        video_tensor = torch.stack(tensors, dim=0)  # (T, C, H, W)
        return video_tensor, torch.tensor(label, dtype=torch.long)
    
    def _extract_frames_from_data(self, video_data) -> List[np.ndarray]:
        """Extract frames from various video data formats"""
        frames = []
        
        # Handle different video data formats
        if isinstance(video_data, bytes):
            # Binary video data
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                tmp.write(video_data)
                tmp_path = tmp.name
            try:
                frames = read_video_frames(
                    Path(tmp_path),
                    frames_per_video=self.frames_per_video,
                    frame_stride=self.frame_stride,
                    face_cropper=None,  # Handle face cropping separately
                )
            finally:
                try:
                    import os
                    os.unlink(tmp_path)
                except:
                    pass
        
        elif isinstance(video_data, dict) and 'bytes' in video_data:
            # HuggingFace Audio/Video feature format
            return self._extract_frames_from_data(video_data['bytes'])
        
        elif isinstance(video_data, str):
            # Video file path
            try:
                frames = read_video_frames(
                    Path(video_data),
                    frames_per_video=self.frames_per_video,
                    frame_stride=self.frame_stride,
                    face_cropper=None,
                )
            except Exception as e:
                print(f"Failed to read video from path {video_data}: {e}")
                frames = []
        
        else:
            print(f"Unknown video data format: {type(video_data)}")
            frames = []
        
        if not frames:
            raise RuntimeError("No frames extracted from video data")
        
        return frames
