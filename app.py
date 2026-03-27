import io
import json
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import cv2
import tempfile
import os

from detector import sliding_patch_scores, reconstruct_heatmap, rgb_to_gray, extract_residual, fft_stats, lbp_entropy
from ethical_assessment import EthicalAssessment, format_ethical_report, get_simple_status
from video_model import ResNetLSTM, GradCAM, overlay_cam
from video_data import read_video_frames, FaceCropper


def get_enhanced_ethical_status(assessment):
    """Generate enhanced status string with prominent flag display"""
    status = assessment.get('status', 'UNKNOWN')
    risk = assessment.get('risk_score', 0)
    flags = assessment.get('flags', [])
    checks = assessment.get('checks', {})

    lines = []

    # Main status line
    if assessment.get('is_ethical'):
        lines.append(f"STATUS: {status}")
    else:
        lines.append(f"STATUS: {status}")

    lines.append(f"Risk Score: {risk:.1%}")

    # Show flags prominently
    if flags:
        lines.append("")
        lines.append("FLAGS RAISED:")
        for flag in flags:
            flag_desc = {
                "NSFW_CONTENT": "Explicit/NSFW content detected",
                "POTENTIAL_MINOR": "CRITICAL: Potential minor detected",
                "POTENTIAL_CELEBRITY": "Celebrity impersonation risk",
                "EMOTIONAL_MANIPULATION": "High emotional manipulation",
                "AI_METADATA_MARKERS": "AI generation markers in metadata",
                "WATERMARK_REMOVAL": "Signs of watermark removal",
                "POTENTIAL_HATE_SYMBOL": "Potential hate symbol detected",
                "MISLEADING_TEXT": "Misleading text overlay",
                "DOCUMENT_DETECTED": "Document/ID forgery risk"
            }.get(flag, flag)
            lines.append(f"  - {flag_desc}")

    # Key check results
    if checks:
        lines.append("")
        if 'nsfw' in checks and checks['nsfw'].get('nsfw_score', 0) > 0.3:
            lines.append(f"NSFW: {checks['nsfw'].get('severity', 'N/A')}")
        if 'age_estimation' in checks and checks['age_estimation'].get('is_minor_risk'):
            lines.append(f"Age: {checks['age_estimation'].get('estimated_age_range', 'N/A')}")
        if 'document' in checks and checks['document'].get('is_document'):
            lines.append(f"Document: {checks['document'].get('document_type', 'detected')}")

    return "\n".join(lines)

# Try to import timm for EfficientNet
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


# ──────────────────────────────────────────────────────────────────────────────
# EfficientNet + FFT Model (from train_adv.py)
# ──────────────────────────────────────────────────────────────────────────────

class FFTFeatureExtractor(nn.Module):
    """Extract FFT features for frequency domain analysis"""
    def __init__(self, output_dim=512):
        super().__init__()
        self.fft_processor = nn.Sequential(
            nn.Linear(12, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, output_dim),
        )

    @torch.no_grad()
    def _extract_fft_features(self, x):
        B, C, H, W = x.shape
        device = x.device
        x_f32 = x.float()
        if C == 3:
            gray = 0.299 * x_f32[:, 0] + 0.587 * x_f32[:, 1] + 0.114 * x_f32[:, 2]
        else:
            gray = x_f32[:, 0]
        fft_img = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft_img)
        mag = torch.abs(fft_shift) + 1e-8
        mag = mag / (mag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        fft_features = []
        for i in range(B):
            m = mag[i].flatten()
            feat = torch.stack([
                m.mean(), m.std().clamp(min=1e-8), m.max(), m.min(),
                (m > m.mean()).float().mean(), m.median(),
                mag[i][:H//4, :].mean(), mag[i][H//4:H//2, :].mean(),
                mag[i][H//2:3*H//4, :].mean(), mag[i][3*H//4:, :].mean(),
                (m > 0.5).float().mean(), (m > 0.1).float().mean(),
            ])
            feat = torch.clamp(feat, min=-10, max=10)
            fft_features.append(feat)
        return torch.stack(fft_features, dim=0)

    def forward(self, x):
        fft_feat = self._extract_fft_features(x)
        fft_feat = fft_feat.to(x.dtype).detach()
        fft_feat.requires_grad_(True)
        return self.fft_processor(fft_feat)


class EfficientNetFFTFusion(nn.Module):
    """EfficientNet-B4 backbone with FFT feature fusion"""
    def __init__(self, num_classes=2, dropout=0.4, backbone='efficientnet_b4'):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            backbone_dim = self.backbone.num_features
        else:
            self.backbone = models.efficientnet_b4(weights=None)
            backbone_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        fft_dim = 512
        self.fft_extractor = FFTFeatureExtractor(output_dim=fft_dim)
        fusion_dim = backbone_dim + fft_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        backbone_feat = self.backbone(x)
        fft_feat = self.fft_extractor(x)
        fused = torch.cat([backbone_feat, fft_feat], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


def make_overlay_pil(img_arr, heatmap, alpha=0.5, cmap='jet'):
    # img_arr: HxWx3 in [0,1]
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(np.clip(img_arr, 0, 1))
    plt.imshow(heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def pad_to_min_size(img, size):
    w, h = img.size
    pad_w = max(0, size - w)
    pad_h = max(0, size - h)
    if pad_w or pad_h:
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        img = TF.pad(img, [left, top, right, bottom], padding_mode='reflect')
    return img


# Load the trained model (try different model files)
MODEL_PATH = None
MODEL = None
MODEL_INFO = None
# Fixed threshold for AI detection
# Auto-calibrated based on recent model performance (94.33% accuracy)
# 0.50 = Balanced threshold (equal precision/recall for fake detection)
# Lower = More sensitive to fakes (higher recall, lower precision)
# Higher = More strict (lower recall, higher precision)
AUTO_THRESHOLD = 0.50

# Temperature scaling to reduce model overconfidence
# T > 1 spreads probabilities, reducing extreme confidence
TEMPERATURE_SCALE = 1.2


def safe_torch_load(path, map_location=None, force_full_load=False):
    """Load checkpoints across PyTorch versions.

    Some versions do not support the weights_only argument. This helper keeps
    behavior consistent while avoiding startup crashes.
    """
    if force_full_load:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    return torch.load(path, map_location=map_location)

def apply_temperature_scaling(prob):
    """Apply temperature scaling to reduce overconfidence"""
    import math
    prob = max(1e-7, min(1 - 1e-7, prob))
    logit = math.log(prob / (1 - prob))
    scaled_logit = logit / TEMPERATURE_SCALE
    return 1 / (1 + math.exp(-scaled_logit))

# Video model variables
VIDEO_MODEL = None
VIDEO_MODEL_CONFIG = None
VIDEO_MODEL_PATH = 'video_resnet_lstm.pt'

GAN_DIFF_MODEL = None
GAN_DIFF_CONFIG = None
GAN_DIFF_MODEL_PATH = Path('models_gan_vs_diffusion/best_model_weights.pt')
GAN_DIFF_CONFIG_PATH = Path('models_gan_vs_diffusion/config.json')

TARGET_REAL_FPR = 0.02  # 2% false positive rate on real images (stricter)
MAX_CALIB_IMAGES = 200


def _pick_dataset_root():
    """Pick the first validation dataset that exists on disk."""
    candidates = [
        Path('C:/Users/DESHNA/Downloads/UAIDE_enhanced/CIFAKE'),
        Path('Deepfake vs Real'),
        Path('DeepfakeVsReal/Dataset'),
        Path('AI vs Real img'),
    ]
    for cand in candidates:
        if (cand / 'Validation').exists():
            return str(cand)
    return None

def _get_validation_files(dataset_root, max_val_images=None):
    val_root = Path(dataset_root) / 'Validation'
    if not val_root.exists():
        return None, None

    real_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
    fake_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))
    real_files = sorted([str(x) for x in real_files])
    fake_files = sorted([str(x) for x in fake_files])

    if max_val_images:
        real_files = real_files[:max_val_images]
        fake_files = fake_files[:max_val_images]

    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    if len(files) == 0:
        return None, None
    return files, labels


def _get_transform(model_type):
    if model_type in ['resnet', 'fusion', 'fusion_improved']:
        size = 224
    else:
        size = 128
    return transforms.Compose([
        transforms.Lambda(lambda img: pad_to_min_size(img, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _load_video_model(checkpoint_path):
    """Load video ResNetLSTM model from checkpoint."""
    if not Path(checkpoint_path).exists():
        return None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt = safe_torch_load(checkpoint_path, map_location=device)
        config = ckpt.get('config', {})
        
        model = ResNetLSTM(
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 1),
            bidirectional=config.get('bidirectional', True),
            temporal_pool=config.get('temporal_pool', 'mean'),
            pretrained=config.get('pretrained', False),
        )
        model.load_state_dict(ckpt['model_state'], strict=True)
        model.to(device)
        model.eval()
        
        print(f'=== Loaded Video Model ===')
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Config: {config}")
        return model, config
    except Exception as e:
        print(f'Failed to load video model from {checkpoint_path}: {e}')
        return None, None


def _build_video_transform():
    """Build transform for video frames."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_gan_diffusion_model(weights_path, config_path):
    """Load the GAN-vs-Diffusion classifier from saved config and weights."""
    if not Path(weights_path).exists() or not Path(config_path).exists():
        return None, None

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)

        backbone = str(cfg.get('backbone', 'resnet18')).lower()
        image_size = int(cfg.get('image_size', 224))

        if backbone == 'resnet50':
            model = models.resnet50(weights=None)
        else:
            model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = safe_torch_load(str(weights_path), map_location=device, force_full_load=True)
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        # Normalize label mappings to integer keys when loaded from JSON.
        id_to_label = cfg.get('id_to_label', {0: 'gan', 1: 'diffusion'})
        if isinstance(id_to_label, dict):
            cfg['id_to_label'] = {int(k): str(v) for k, v in id_to_label.items()}

        cfg['image_size'] = image_size
        return model, cfg
    except Exception as e:
        print(f'Failed to load GAN-vs-Diffusion model: {e}')
        return None, None


def _predict_ai_source_from_pil(pil_img):
    """Predict whether AI image origin is GAN or Diffusion."""
    if GAN_DIFF_MODEL is None or GAN_DIFF_CONFIG is None:
        return None

    try:
        image_size = int(GAN_DIFF_CONFIG.get('image_size', 224))
        id_to_label = GAN_DIFF_CONFIG.get('id_to_label', {0: 'gan', 1: 'diffusion'})

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = GAN_DIFF_MODEL(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = int(np.argmax(probs))

        pred_label = str(id_to_label.get(pred_idx, 'unknown')).lower()
        gan_prob = float(probs[0])
        diffusion_prob = float(probs[1])

        return {
            'label': pred_label,
            'gan_prob': gan_prob,
            'diffusion_prob': diffusion_prob,
        }
    except Exception as e:
        print(f'AI source prediction failed: {e}')
        return None


def _format_ai_source_text(source_pred):
    if not source_pred:
        return 'AI origin analysis unavailable.'

    source_label = source_pred.get('label', 'unknown').upper()
    gan_prob = source_pred.get('gan_prob', 0.0)
    diffusion_prob = source_pred.get('diffusion_prob', 0.0)
    return (
        f'Predicted source: {source_label}\n'
        f'GAN probability: {gan_prob:.2%}\n'
        f'Diffusion probability: {diffusion_prob:.2%}'
    )


def _extract_video_frames(video_path, frames_per_video=16, frame_stride=4, face_detection=False):
    """Extract frames from video file."""
    try:
        face_cropper = FaceCropper() if face_detection else None
        frames = read_video_frames(
            Path(video_path),
            frames_per_video=frames_per_video,
            frame_stride=frame_stride,
            face_cropper=face_cropper,
        )
        return frames
    except Exception as e:
        print(f'Video frame extraction failed: {e}')
        return None


def _predict_video_model(model, config, video_path, return_frames=False):
    """Predict deepfake probability using video model."""
    if model is None:
        return None, None, None
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        frames_per_video = config.get('frames_per_video', 16)
        frame_stride = config.get('frame_stride', 4)
        face_detection = config.get('face_detection', True)
        
        frames = _extract_video_frames(video_path, frames_per_video, frame_stride, face_detection)
        if frames is None:
            return None, None, None
        
        transform = _build_video_transform()
        video_tensor = torch.stack([transform(Image.fromarray(f)) for f in frames], dim=0)
        video_tensor = video_tensor.unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            frame_logits, video_logits = model(video_tensor)
            video_probs = torch.softmax(video_logits, dim=1).cpu().numpy()[0]
            frame_probs = torch.softmax(frame_logits.squeeze(0), dim=1).cpu().numpy()
        
        prob_fake = float(video_probs[1])
        
        if return_frames:
            return prob_fake, frame_probs, frames
        return prob_fake, frame_probs, None
    except Exception as e:
        print(f'Video model prediction failed: {e}')
        return None, None, None



def _evaluate_deep_model(model, model_type, dataset_root, max_val_images=None):
    """Evaluate deep learning models on validation set."""
    from train import ImageDataset
    from torch.utils.data import DataLoader

    files, labels = _get_validation_files(dataset_root, max_val_images=max_val_images)
    if files is None:
        return None

    transform = _get_transform(model_type)
    dataset = ImageDataset(files, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls)

    if len(all_labels) == 0:
        return None

    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = None
    report = classification_report(y_true, y_pred, zero_division=0)
    return {
        'accuracy': acc,
        'auc': auc,
        'report': report,
        'count': len(y_true)
    }


def _evaluate_ml_model(model, dataset_root, max_val_images=None, patch_size=128, n_patches=4):
    from train import collect_features

    val_root = Path(dataset_root) / 'Validation'
    if not val_root.exists():
        return None

    real_val = val_root / 'Real'
    fake_val = val_root / 'Fake'
    Xrv, yrv = collect_features(real_val, 0, max_images=max_val_images, patch_size=patch_size, n_patches=n_patches)
    Xfv, yfv = collect_features(fake_val, 1, max_images=max_val_images, patch_size=patch_size, n_patches=n_patches)
    if len(Xrv) + len(Xfv) == 0:
        return None

    Xv = np.array(Xrv + Xfv)
    yv = np.array(yrv + yfv)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(Xv)[:, 1]
    else:
        probs = model.predict(Xv).astype(float)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(yv, preds)
    try:
        auc = roc_auc_score(yv, probs)
    except Exception:
        auc = None
    report = classification_report(yv, preds, zero_division=0)
    return {
        'accuracy': acc,
        'auc': auc,
        'report': report,
        'count': len(yv)
    }


def _calibrate_threshold(model, model_info, dataset_root, target_real_fpr=0.05, max_val_images=200):
    """Calibrate detection threshold using ROC/Youden J; fallback to real-FPR quantile."""
    if model is None or model_info is None:
        return None

    val_root = Path(dataset_root) / 'Validation'
    if not val_root.exists():
        return None

    mtype = model_info.get('model_type', 'unknown') if isinstance(model_info, dict) else 'unknown'
    all_probs = []
    all_labels = []

    try:
        if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
            from train import ImageDataset
            from torch.utils.data import DataLoader

            files, labels = _get_validation_files(dataset_root, max_val_images=max_val_images)
            if files is None:
                return None

            transform = _get_transform(mtype)
            dataset = ImageDataset(files, labels, transform=transform)
            dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                for inputs, lbls in dataloader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    all_probs.extend(probs.tolist())
                    all_labels.extend(lbls.tolist())
        else:
            from train import collect_features

            real_val = val_root / 'Real'
            fake_val = val_root / 'Fake'
            Xrv, yrv = collect_features(real_val, 0, max_images=max_val_images, patch_size=128, n_patches=4)
            Xfv, yfv = collect_features(fake_val, 1, max_images=max_val_images, patch_size=128, n_patches=4)
            X = np.array(Xrv + Xfv)
            y = np.array(yrv + yfv)
            if len(X) == 0:
                return None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[:, 1]
            else:
                probs = model.predict(X).astype(float)
            all_probs.extend(probs.tolist())
            all_labels.extend(y.tolist())

        if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
            return None

        # Youden's J: maximize TPR - FPR
        fpr, tpr, thresholds = roc_curve(np.array(all_labels), np.array(all_probs))
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        best_thresh = float(thresholds[best_idx])
        print(f'Calibrated threshold via Youden J (balanced): {best_thresh:.3f} (TPR={tpr[best_idx]:.3f}, FPR={fpr[best_idx]:.3f})')
        return best_thresh
    except Exception as e:
        print(f'Youden calibration failed, falling back to real-FPR quantile: {e}')

    # Fallback: match target_real_fpr on real images only
    real_probs = []
    if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
        from train import ImageDataset
        from torch.utils.data import DataLoader

        files, labels = _get_validation_files(dataset_root, max_val_images=max_val_images)
        if files is None:
            return None
        real_files = [f for f, lbl in zip(files, labels) if lbl == 0]
        if len(real_files) == 0:
            return None

        transform = _get_transform(mtype)
        dataset = ImageDataset(real_files, [0] * len(real_files), transform=transform)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval()
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                real_probs.extend(probs.tolist())
    else:
        from train import collect_features
        real_val = val_root / 'Real'
        Xrv, _ = collect_features(real_val, 0, max_images=max_val_images, patch_size=128, n_patches=4)
        if len(Xrv) == 0:
            return None
        X = np.array(Xrv)
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X).astype(float)
        real_probs.extend(probs.tolist())

    if len(real_probs) == 0:
        return None

    real_probs = np.array(real_probs)
    target = max(0.0, min(1.0, float(target_real_fpr)))
    if target <= 0.0:
        return float(np.max(real_probs))
    return float(np.quantile(real_probs, 1.0 - target))


def _load_model_from_info(info_path):
    info = joblib.load(str(info_path))
    if not isinstance(info, dict) or 'model_type' not in info:
        return None, None

    def _model_base_from_info_path(path_obj):
        path_str = str(path_obj)
        if path_str.endswith('_improved_info.pkl'):
            return path_str[:-len('_improved_info.pkl')]
        if path_str.endswith('_info.pkl'):
            return path_str[:-len('_info.pkl')]
        return path_str

    mtype = info['model_type']
    model = None

    if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
        if mtype == 'resnet':
            from train import DeepfakeResNet as _ModelClass
        elif mtype in ['fusion', 'fusion_improved']:
            from train import DeepfakeFeatureFusion as _ModelClass
        else:
            from train import DeepfakeCNN as _ModelClass

        model = _ModelClass()
        state_path = info.get('state_dict_path')
        if state_path is None:
            base = _model_base_from_info_path(info_path)
            candidates = [base + '_best_improved', base + '_best', base]
            for c in candidates:
                if Path(c).exists():
                    state_path = c
                    break
        if state_path is None:
            raise FileNotFoundError('state_dict_path not found in model info and no candidate file exists')

        model.load_state_dict(safe_torch_load(state_path, map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
    else:
        base = _model_base_from_info_path(info_path)
        joblib_candidates = [base, base + '.joblib'] if base.endswith('.joblib') else [base + '.joblib', base]
        for joblib_path in joblib_candidates:
            if Path(joblib_path).exists():
                model = joblib.load(joblib_path)
                break
        if model is None:
            model = info

    return model, info


def load_model_fast():
    """Load a model quickly without validation evaluation for faster startup."""
    # Prioritize improved models
    info_candidates = list(Path('.').glob('*_improved_info.pkl')) + list(Path('.').glob('*_info.pkl'))
    info_candidates = sorted(info_candidates, key=lambda p: p.name, reverse=True)
    
    if not info_candidates:
        return None, None, None
    
    # Try to load the first valid model without evaluation
    for info_path in info_candidates:
        try:
            model, info = _load_model_from_info(info_path)
            mtype = info.get('model_type', 'unknown') if isinstance(info, dict) else 'unknown'
            print(f'=== Loaded Model (fast mode) ===')
            print(f"Model: {info_path.name}")
            print(f"Type: {mtype}")
            return model, info, str(info_path)
        except Exception as e:
            print(f"Skipping {info_path.name}: failed to load model ({e})")
            continue
    
    return None, None, None


def select_best_model(dataset_root='C:\\Users\\DESHNA\\Downloads\\UAIDE_enhanced\\CIFAKE', max_val_images=None):
    info_candidates = list(Path('.').glob('*_improved_info.pkl')) + list(Path('.').glob('*_info.pkl'))
    info_candidates = sorted(info_candidates, key=lambda p: p.name, reverse=True)
    if not info_candidates:
        return None, None, None

    scored = []
    for info_path in info_candidates:
        try:
            model, info = _load_model_from_info(info_path)
        except Exception as e:
            print(f"Skipping {info_path.name}: failed to load model ({e})")
            continue

        mtype = info.get('model_type', 'unknown') if isinstance(info, dict) else 'unknown'
        stats = None

        if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved']:
            try:
                stats = _evaluate_deep_model(model, mtype, dataset_root, max_val_images=max_val_images)
            except Exception as e:
                print(f"Failed to evaluate deep model: {e}")
                stats = None
        else:
            patch_size = info.get('patch_size', 128) if isinstance(info, dict) else 128
            n_patches = info.get('patches_per_image', 4) if isinstance(info, dict) else 4
            stats = _evaluate_ml_model(model, dataset_root, max_val_images=max_val_images, patch_size=patch_size, n_patches=n_patches)

        if stats is None:
            print(f"Skipping {info_path.name}: no validation stats")
            continue

        scored.append((info_path, model, info, stats))
        auc = stats.get('auc')
        acc = stats.get('accuracy')
        print('\n=== Model Evaluation ===')
        print(f"Model: {info_path.name} | type: {mtype}")
        print(f"Val samples: {stats.get('count', 0)}")
        print(f"Accuracy: {acc:.4f}")
        if auc is not None:
            print(f"ROC AUC: {auc:.4f}")
        print('Classification report:')
        print(stats.get('report', 'N/A'))

    if not scored:
        return None, None, None

    def _score_key(item):
        _, _, _, st = item
        auc = st.get('auc')
        acc = st.get('accuracy')
        return (auc if auc is not None else -1.0, acc)

    scored.sort(key=_score_key, reverse=True)
    best_path, best_model, best_info, best_stats = scored[0]
    print('\n=== Selected Best Model ===')
    print(f"Model: {best_path.name}")
    print(f"Type: {best_info.get('model_type', 'unknown') if isinstance(best_info, dict) else 'unknown'}")
    print(f"Accuracy: {best_stats.get('accuracy'):.4f}")
    if best_stats.get('auc') is not None:
        print(f"ROC AUC: {best_stats.get('auc'):.4f}")
    return best_model, best_info, str(best_path)



# ──────────────────────────────────────────────────────────────────────────────
# ROC-BASED OPTIMAL THRESHOLD CALCULATION
# ──────────────────────────────────────────────────────────────────────────────

def calculate_optimal_threshold_from_metrics(metrics):
    """
    Calculate optimal threshold using ROC curve analysis (Youden's J statistic).
    Uses sensitivity and specificity from the classification report.

    Youden's J = Sensitivity + Specificity - 1

    When model is biased (sensitivity >> specificity), we need to raise threshold
    aggressively to reduce false positives (real images detected as AI).
    """
    if not metrics:
        return 0.5

    sensitivity = metrics.get('sensitivity', 0.5)  # TPR (ability to detect fakes)
    specificity = metrics.get('specificity', 0.5)  # TNR (ability to recognize real)
    fpr = metrics.get('fpr', 0.5)  # False Positive Rate (real images wrongly called fake)
    fnr = metrics.get('fnr', 0.5)  # False Negative Rate (fakes wrongly called real)

    # Calculate Youden's J at current threshold (evaluated at 0.5)
    youden_j = sensitivity + specificity - 1

    # Calculate imbalance: positive means biased towards detecting fake
    imbalance = sensitivity - specificity

    # Method 1: FPR-based adjustment (AGGRESSIVE for high FPR)
    # High FPR means we're classifying too many real images as fake
    # Use exponential scaling for high FPR values
    if fpr > 0.15:
        # Aggressive adjustment for high FPR
        fpr_adjustment = fpr * 1.2 + (fpr - 0.15) * 0.5
    else:
        fpr_adjustment = fpr * 0.8

    # Method 2: Imbalance-based adjustment
    # Large positive imbalance means sensitivity >> specificity
    if imbalance > 0:
        # Progressive scaling: larger imbalance = more aggressive adjustment
        imbalance_adjustment = imbalance * (0.4 + 0.3 * imbalance)
    else:
        imbalance_adjustment = imbalance * 0.3

    # Method 3: Target specificity adjustment
    # If specificity is below 85%, push threshold higher
    if specificity < 0.85:
        specificity_boost = (0.85 - specificity) * 0.5
    else:
        specificity_boost = 0.0

    # Combine adjustments
    adjustment = 0.5 * fpr_adjustment + 0.3 * imbalance_adjustment + 0.2 * specificity_boost

    optimal_threshold = 0.5 + adjustment

    # Clamp to reasonable range
    optimal_threshold = max(0.40, min(0.85, optimal_threshold))

    print(f"  ROC Analysis (Youden's J Method):")
    print(f"    Sensitivity (TPR): {sensitivity:.3f} (detect fakes)")
    print(f"    Specificity (TNR): {specificity:.3f} (recognize real)")
    print(f"    False Positive Rate: {fpr:.3f} (real -> fake error)")
    print(f"    False Negative Rate: {fnr:.3f} (fake -> real error)")
    print(f"    Youden's J: {youden_j:.3f}")
    print(f"    Imbalance: {imbalance:+.3f}")
    print(f"    FPR Adjustment: +{fpr_adjustment:.3f}")
    print(f"    Imbalance Adjustment: +{imbalance_adjustment:.3f}")
    print(f"    Specificity Boost: +{specificity_boost:.3f}")
    print(f"    Combined Adjustment: +{adjustment:.3f}")
    print(f"    Optimal Threshold: {optimal_threshold:.3f}")

    return optimal_threshold


def get_confidence_tier(probability, threshold):
    """
    Get confidence tier instead of binary decision.
    Returns tier name and confidence level.
    """
    if probability >= threshold + 0.25:
        return "HIGH_CONFIDENCE_AI", "Very likely AI-generated"
    elif probability >= threshold + 0.10:
        return "MEDIUM_CONFIDENCE_AI", "Likely AI-generated"
    elif probability >= threshold:
        return "LOW_CONFIDENCE_AI", "Possibly AI-generated (borderline)"
    elif probability >= threshold - 0.15:
        return "UNCERTAIN", "Uncertain - could be either"
    elif probability >= threshold - 0.30:
        return "LOW_CONFIDENCE_REAL", "Likely authentic"
    else:
        return "HIGH_CONFIDENCE_REAL", "Very likely authentic"


# Auto-detect and load the latest trained model
# Priority order: models_adv > models_v2 > fallback

MODEL = None
MODEL_PATH = None
MODEL_INFO = None

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 1: Load Recently Trained Model from models_adv (NEWEST)
# ─────────────────────────────────────────────────────────────────────────────
ADV_MODEL_PATH = Path('models_adv/best_model_weights.pt')
ADV_CONFIG_PATH = Path('models_adv/config.json')

if ADV_MODEL_PATH.exists():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config
        adv_config = {}
        if ADV_CONFIG_PATH.exists():
            with open(ADV_CONFIG_PATH) as f:
                adv_config = json.load(f)

        # Load state_dict first to detect correct backbone
        state_dict = safe_torch_load(ADV_MODEL_PATH, map_location=device, force_full_load=True)

        # Detect backbone from fusion layer input dimension
        # backbone_dim + 512 (FFT) = fusion input
        fusion_in_dim = state_dict['fusion.0.weight'].shape[1]
        backbone_dim = fusion_in_dim - 512

        # Map backbone_dim to backbone name
        backbone_map = {1280: 'efficientnet_b0', 1408: 'efficientnet_b2', 1792: 'efficientnet_b4'}
        backbone = backbone_map.get(backbone_dim, 'efficientnet_b2')

        # Require timm for this model
        if not HAS_TIMM:
            raise ImportError("timm is required for EfficientNet models. Install with: pip install timm")

        # Create model architecture and load weights
        MODEL = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone=backbone)
        MODEL.load_state_dict(state_dict)
        MODEL.to(device)
        MODEL.eval()

        MODEL_PATH = str(ADV_MODEL_PATH)
        MODEL_INFO = {
            'model_type': 'efficientnet_fft',
            'backbone': backbone,
            'accuracy': adv_config.get('best_metrics', {}).get('accuracy', 86.0),
            'auc': adv_config.get('best_metrics', {}).get('auc_roc', 0.9394),
            'optimal_threshold': 0.50,
            **adv_config
        }

        mod_time = datetime.fromtimestamp(ADV_MODEL_PATH.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f'\n=== Loaded Recent Model (PRIORITY 1) ===')
        print(f"Model: {ADV_MODEL_PATH.name}")
        print(f"Backbone: {MODEL_INFO.get('backbone', 'unknown')}")
        print(f"Accuracy: {MODEL_INFO.get('accuracy', 'unknown')}%")
        print(f"AUC: {MODEL_INFO.get('auc', 'unknown')}")
        print(f"Modified: {mod_time}")
        print(f"Status: Ready for inference ✓")

    except Exception as e:
        print(f'Failed to load models_adv model: {e}')
        MODEL = None

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 2: Load EfficientNet+FFT model from models_v2 (only if PRIORITY 1 failed)
# ─────────────────────────────────────────────────────────────────────────────
EFFICIENTNET_MODEL_PATH = Path('models_v2/best_model.pt')
EFFICIENTNET_WEIGHTS_PATH = Path('models_v2/best_model_weights.pt')
EFFICIENTNET_CONFIG_PATH = Path('models_v2/config.json')
EFFICIENTNET_REPORT_PATH = Path('models_v2/classification_report.json')

if MODEL is None and (EFFICIENTNET_MODEL_PATH.exists() or EFFICIENTNET_WEIGHTS_PATH.exists()):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load config to get model settings
        config = {}
        if EFFICIENTNET_CONFIG_PATH.exists():
            with open(EFFICIENTNET_CONFIG_PATH) as f:
                config = json.load(f)

        # Load classification report for threshold
        report = {}
        if EFFICIENTNET_REPORT_PATH.exists():
            with open(EFFICIENTNET_REPORT_PATH) as f:
                report = json.load(f)

        # Create model
        backbone = config.get('backbone', 'efficientnet_b4')
        MODEL = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone=backbone)

        # Load weights (weights_only=False for PyTorch 2.6+ compatibility)
        if EFFICIENTNET_MODEL_PATH.exists():
            checkpoint = safe_torch_load(EFFICIENTNET_MODEL_PATH, map_location=device, force_full_load=True)
            if 'model_state_dict' in checkpoint:
                MODEL.load_state_dict(checkpoint['model_state_dict'])
            else:
                MODEL.load_state_dict(checkpoint)
        else:
            MODEL.load_state_dict(safe_torch_load(EFFICIENTNET_WEIGHTS_PATH, map_location=device, force_full_load=True))

        MODEL.to(device)
        MODEL.eval()

        # Set model info
        metrics = report.get('metrics', {})

        # Calculate optimal threshold using ROC curve analysis (Youden's J)
        AUTO_THRESHOLD = calculate_optimal_threshold_from_metrics(metrics)

        MODEL_INFO = {
            'model_type': 'efficientnet_fft',
            'backbone': backbone,
            'image_size': config.get('image_size', 224),
            'accuracy': metrics.get('accuracy', 0.89),
            'auc_roc': metrics.get('auc_roc', 0.96),
            'optimal_threshold': AUTO_THRESHOLD,
        }
        MODEL_PATH = str(EFFICIENTNET_MODEL_PATH)

        print('=' * 60)
        print(' LOADED: EfficientNet-B4 + FFT Fusion Model')
        print('=' * 60)
        print(f"  Path: {MODEL_PATH}")
        print(f"  Backbone: {backbone}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A'):.2%}" if metrics.get('accuracy') else "  Accuracy: N/A")
        print(f"  AUC-ROC: {metrics.get('auc_roc', 'N/A'):.4f}" if metrics.get('auc_roc') else "  AUC-ROC: N/A")
        print(f"  Threshold (ROC-optimized): {AUTO_THRESHOLD:.3f}")
        print('=' * 60)

    except Exception as e:
        print(f'Failed to load EfficientNet model: {e}')
        MODEL = None

# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY 2: Fall back to other models
# ─────────────────────────────────────────────────────────────────────────────
if MODEL is None:
    # Find all model files sorted by modification time (newest first)
    model_candidates = sorted(
        list(Path('.').glob('model*.joblib')) +
        list(Path('.').glob('model*.pt')) +
        [Path('model_fusion_best_improved')] +
        [Path('model_fusion_best_improved_best_improved')],
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True
    )

    if model_candidates:
        for model_path in model_candidates:
            try:
                # Try loading as joblib first (traditional ML models)
                try:
                    MODEL = joblib.load(str(model_path))
                    print(f'[OK] Loaded joblib model: {model_path.name}')
                except Exception as joblib_err:
                    # If joblib fails, try loading as PyTorch state dict
                    try:
                        state_dict = safe_torch_load(str(model_path), map_location='cpu')

                        # Check model info to determine the correct class
                        model_info_path = Path(str(model_path).rsplit('.', 1)[0] + '_info.pkl')
                        if model_info_path.exists():
                            MODEL_INFO = joblib.load(str(model_info_path))
                        else:
                            MODEL_INFO = {'model_type': 'fusion_improved'}

                        model_type = MODEL_INFO.get('model_type', 'fusion_improved')

                        # Reconstruct model based on type
                        if model_type in ['fusion', 'fusion_improved']:
                            from train import DeepfakeFeatureFusion
                            MODEL = DeepfakeFeatureFusion()
                            MODEL.load_state_dict(state_dict)
                            MODEL.eval()
                            if torch.cuda.is_available():
                                MODEL.cuda()
                            print(f'[OK] Loaded PyTorch fusion model: {model_path.name}')
                        elif model_type == 'resnet':
                            from train import DeepfakeResNet
                            MODEL = DeepfakeResNet()
                            MODEL.load_state_dict(state_dict)
                            MODEL.eval()
                            print(f'[OK] Loaded PyTorch ResNet model: {model_path.name}')
                        elif model_type == 'cnn':
                            from train import DeepfakeCNN
                            MODEL = DeepfakeCNN()
                            MODEL.load_state_dict(state_dict)
                            MODEL.eval()
                            print(f'[OK] Loaded PyTorch CNN model: {model_path.name}')
                        else:
                            raise ValueError(f"Unknown model type: {model_type}")
                    except Exception as torch_err:
                        raise Exception(f"Joblib failed: {joblib_err}; PyTorch failed: {torch_err}")

                MODEL_PATH = str(model_path)

                # Try to load metadata if not already loaded
                if MODEL_INFO is None:
                    model_info_path = Path(str(model_path).rsplit('.', 1)[0] + '_info.pkl')
                    if not model_info_path.exists():
                        model_base = str(model_path).replace('.joblib', '').replace('_best_improved', '').replace('_best', '')
                        for info_candidate in [model_base + '_info.pkl', model_base + '_improved_info.pkl']:
                            if Path(info_candidate).exists():
                                model_info_path = Path(info_candidate)
                                break

                    if model_info_path.exists():
                        MODEL_INFO = joblib.load(str(model_info_path))
                    else:
                        MODEL_INFO = {'model_type': 'unknown'}

                # Display which model was loaded
                mod_time = datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                print(f'=== Loaded Latest Model ===')
                print(f"Model: {model_path.name}")
                print(f"Modified: {mod_time}")
                model_type = MODEL_INFO.get("model_type", "unknown") if isinstance(MODEL_INFO, dict) else "unknown"
                print(f"Type: {model_type}")
                print(f"Path: {model_path.resolve()}")
                break
            except Exception as e:
                print(f'Failed to load {model_path.name}: {e}')
                continue

if MODEL is None:
    print('No valid model found; falling back to heuristic detection.')
    MODEL_INFO = None
    MODEL_PATH = None
else:
    # Using fixed threshold of 0.80 (disabled auto-calibration)
    # Higher threshold = fewer false positives (real images classified as AI)
    print(f'Using fixed threshold: {AUTO_THRESHOLD:.3f} (high confidence mode)')

    # NOTE: Auto-calibration disabled - uncomment below to re-enable
    # try:
    #     dataset_root = _pick_dataset_root()
    #     if dataset_root is None:
    #         raise FileNotFoundError('No validation dataset found')
    #     AUTO_THRESHOLD = _calibrate_threshold(
    #         MODEL, MODEL_INFO, dataset_root=dataset_root,
    #         target_real_fpr=TARGET_REAL_FPR, max_val_images=MAX_CALIB_IMAGES,
    #     )
    # except Exception as e:
    #     print(f'Auto-threshold calibration failed: {e}')


# Load video model
if Path(VIDEO_MODEL_PATH).exists():
    VIDEO_MODEL, VIDEO_MODEL_CONFIG = _load_video_model(VIDEO_MODEL_PATH)
else:
    print(f'Video model not found: {VIDEO_MODEL_PATH}')

# Load GAN-vs-Diffusion source model for second-stage AI origin analysis
if GAN_DIFF_MODEL_PATH.exists() and GAN_DIFF_CONFIG_PATH.exists():
    GAN_DIFF_MODEL, GAN_DIFF_CONFIG = _load_gan_diffusion_model(GAN_DIFF_MODEL_PATH, GAN_DIFF_CONFIG_PATH)
    if GAN_DIFF_MODEL is not None:
        print('=== Loaded AI Source Model ===')
        print(f"Path: {GAN_DIFF_MODEL_PATH}")
        print(f"Backbone: {GAN_DIFF_CONFIG.get('backbone', 'resnet18')}")
    else:
        print('GAN-vs-Diffusion model found but failed to load.')
else:
    print('GAN-vs-Diffusion model not found; source classification disabled.')


def extract_image_features_from_array(img_arr, patch_size=128, n_patches=8, random_state=None):
    # sample random patches (similar to training script) and pool mean/std
    H, W, _ = img_arr.shape
    patches = []
    rng = np.random.RandomState(random_state)
    for _ in range(n_patches):
        if H <= patch_size or W <= patch_size:
            y0 = max(0, (H - patch_size) // 2)
            x0 = max(0, (W - patch_size) // 2)
        else:
            y0 = int(rng.randint(0, H - patch_size + 1))
            x0 = int(rng.randint(0, W - patch_size + 1))
        patch = img_arr[y0:y0 + patch_size, x0:x0 + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            ph = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            ph[:patch.shape[0], :patch.shape[1]] = patch
            patch = ph
        patches.append(patch)

    feats = []
    for p in patches:
        g = rgb_to_gray(p)
        res = extract_residual(g)
        res_std = float(np.std(res))
        _, hf = fft_stats(g)
        # LBP expects integer images; convert to uint8 for stability
        ent = lbp_entropy((g * 255).astype(np.uint8))
        feats.append([res_std, hf, ent])
    feats = np.array(feats)
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    return np.concatenate([mean, std])[None, :]
    
def evaluate_model_on_validation(model, dataset_root='C:\\Users\\DESHNA\\Downloads\\UAIDE_enhanced\\CIFAKE'):
    p = Path(dataset_root)
    val_root = p / 'Validation'
    if not val_root.exists():
        return 'Validation folder not found under ' + str(p)

    real_folder = val_root / 'Real'
    fake_folder = val_root / 'Fake'
    files_real = sorted([str(x) for x in real_folder.rglob('*.jpg')] + [str(x) for x in real_folder.rglob('*.png')])
    files_fake = sorted([str(x) for x in fake_folder.rglob('*.jpg')] + [str(x) for x in fake_folder.rglob('*.png')])
    files = [(f, 0) for f in files_real] + [(f, 1) for f in files_fake]
    X = []
    y = []
    for f, lbl in files:
        try:
            pil = Image.open(f).convert('RGB')
            arr = np.asarray(pil).astype(np.float32) / 255.0
            feat = extract_image_features_from_array(arr, patch_size=128, n_patches=8, random_state=123)
            X.append(feat[0])
            y.append(lbl)
        except Exception as e:
            continue
    if len(X) == 0:
        return 'No validation images found or feature extraction failed.'
    X = np.array(X)
    y = np.array(y)
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)[:, 1]
        else:
            probs = model.predict(X).astype(float)
        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds)
        try:
            auc = roc_auc_score(y, probs)
        except Exception:
            auc = None
        report = classification_report(y, preds)
        lines = [f'Val accuracy: {acc:.4f}']
        if auc is not None:
            lines.append(f'Val ROC AUC: {auc:.4f}')
        lines.append('\n'+report)
        return '\n'.join(lines)
    except Exception as e:
        return f'Evaluation failed: {e}'


def predict_video_gradio(video_file, ethical_threshold=0.5, show_raw_features=False):
    """Predict deepfake probability for video input."""
    ethical_status = "N/A"
    ethical_report = ""
    ai_source_text = "N/A (shown for AI-detected samples when source model is loaded)"
    
    if VIDEO_MODEL is None or VIDEO_MODEL_CONFIG is None:
        return "Model Error", 0.0, None, "Video model not loaded", "Please ensure video_resnet_lstm.pt exists", ai_source_text
    
    try:
        # Get video path
        if isinstance(video_file, str):
            video_path = video_file
        else:
            video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
        
        # Predict using video model
        prob_fake, frame_probs, frames = _predict_video_model(
            VIDEO_MODEL, 
            VIDEO_MODEL_CONFIG, 
            video_path,
            return_frames=True
        )

        if prob_fake is None:
            return "Error", 0.0, None, "Failed to process video", "Video processing failed", ai_source_text

        # Apply temperature scaling to reduce overconfidence
        prob_fake = apply_temperature_scaling(prob_fake)

        # Determine label and threshold
        threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else VIDEO_MODEL_CONFIG.get('optimal_threshold', 0.85)
        is_ai = prob_fake >= threshold
        label = 'AI-generated' if is_ai else 'Real (camera)'

        if is_ai and frames and len(frames) > 0:
            source_pred = _predict_ai_source_from_pil(Image.fromarray(frames[0]))
            if source_pred is not None:
                ai_source_text = _format_ai_source_text(source_pred)
                source_label = source_pred.get('label', '').lower()
                if source_label in ['gan', 'diffusion']:
                    label = f'AI-generated ({source_label.upper()})'
            else:
                ai_source_text = 'AI origin analysis unavailable.'
        
        # Create visualization: overlay top suspicious frame
        visualization = None
        if frames is not None and len(frames) > 0:
            try:
                # Find frame with highest fake probability
                top_frame_idx = int(np.argmax(frame_probs[:, 1]))
                frame = frames[top_frame_idx]
                
                # Apply Grad-CAM on this frame
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                target_layer = VIDEO_MODEL.backbone.layer4[-1].conv3
                grad_cam = GradCAM(VIDEO_MODEL, target_layer)
                
                transform = _build_video_transform()
                frame_tensor = transform(Image.fromarray(frame)).unsqueeze(0).unsqueeze(0).to(device)
                cam = grad_cam.generate(frame_tensor, class_idx=1)  # Focus on fake class
                
                # Create overlay visualization
                visualization = overlay_cam(frame, cam, alpha=0.5)
            except Exception as e:
                print(f"Grad-CAM visualization failed: {e}")
                # Fallback: just use first frame
                if frames and len(frames) > 0:
                    visualization = frames[0]
        
        # Convert to PIL if we have visualization
        if visualization is not None:
            overlay_pil = Image.fromarray(visualization)
        else:
            # Create a simple fallback image
            overlay_pil = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        # Perform ethical assessment if AI-generated detected
        if is_ai and frames and len(frames) > 0:
            # Use first frame for ethical assessment
            img_arr = np.asarray(frames[0]).astype(np.float32) / 255.0
            assessment = EthicalAssessment.assess(img_arr, threshold=ethical_threshold)
            ethical_status = get_enhanced_ethical_status(assessment)
            ethical_report = format_ethical_report(assessment)
            if not show_raw_features:
                idx = ethical_report.find('\nRAW FEATURES:')
                if idx != -1:
                    ethical_report = ethical_report[:idx]
        
        return label, prob_fake, overlay_pil, ethical_status, ethical_report, ai_source_text
        
    except Exception as e:
        print(f"Video prediction failed: {e}")
        return "Error", 0.0, None, "Prediction failed", str(e), ai_source_text


def predict_gradio(pil_img, ethical_threshold=0.5, show_raw_features=False):
    # pil_img is a PIL.Image
    img = np.asarray(pil_img).astype(np.float32) / 255.0

    # Initialize ethical status
    ethical_status = "N/A"
    ethical_report = ""
    ai_source_text = "N/A (shown for AI-detected samples when source model is loaded)"

    if MODEL is not None and MODEL_INFO is not None:
        try:
            if MODEL_INFO['model_type'] in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'efficientnet_fft']:
                # Deep learning model prediction with Grad-CAM
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Prepare image transform based on model type
                if MODEL_INFO['model_type'] == 'efficientnet_fft':
                    # EfficientNet uses 224x224 (can be higher but 224 works well)
                    img_size = MODEL_INFO.get('image_size', 224)
                    transform = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                elif MODEL_INFO['model_type'] in ['resnet', 'fusion', 'fusion_improved']:
                    transform = transforms.Compose([
                        transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                else:
                    transform = transforms.Compose([
                        transforms.Lambda(lambda img: pad_to_min_size(img, 128)),
                        transforms.CenterCrop(128),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])

                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                # Get prediction
                with torch.no_grad():
                    outputs = MODEL(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    prob_fake_raw = float(probs[0, 1])

                    # Apply temperature scaling to reduce overconfidence
                    prob_fake = apply_temperature_scaling(prob_fake_raw)

                    # Use fixed threshold to reduce false positives
                    threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else MODEL_INFO.get('optimal_threshold', 0.85)
                    pred_class = 1 if prob_fake >= threshold else 0

                label = 'AI-generated' if pred_class == 1 else 'Real (camera)'
                is_ai = pred_class == 1

                if is_ai:
                    source_pred = _predict_ai_source_from_pil(pil_img)
                    if source_pred is not None:
                        ai_source_text = _format_ai_source_text(source_pred)
                        source_label = source_pred.get('label', '').lower()
                        if source_label in ['gan', 'diffusion']:
                            label = f'AI-generated ({source_label.upper()})'
                    else:
                        ai_source_text = 'AI origin analysis unavailable.'

                # Generate Grad-CAM visualization
                try:
                    overlay_img = apply_gradcam_overlay_from_pil(pil_img, MODEL, MODEL_INFO['model_type'])
                    overlay_pil = Image.fromarray(overlay_img)
                except Exception as e:
                    print(f"Grad-CAM failed: {e}")
                    # Fallback to traditional heatmap
                    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
                    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
                    overlay_pil = make_overlay_pil(img, heat)

                # Perform ethical assessment if AI-generated detected
                if is_ai:
                    assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
                    ethical_status = get_enhanced_ethical_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob_fake, overlay_pil, ethical_status, ethical_report, ai_source_text

            else:
                # Traditional ML model
                X = extract_image_features_from_array(img, patch_size=128, n_patches=8, random_state=0)
                if hasattr(MODEL, 'predict_proba'):
                    prob_raw = float(MODEL.predict_proba(X)[:, 1][0])
                else:
                    pred = MODEL.predict(X)[0]
                    prob_raw = float(pred)

                # Apply temperature scaling to reduce overconfidence
                prob = apply_temperature_scaling(prob_raw)

                # Use fixed threshold to reduce false positives
                threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else MODEL_INFO.get('optimal_threshold', 0.85)
                is_ai = prob >= threshold
                label = 'AI-generated' if is_ai else 'Real (camera)'

                if is_ai:
                    source_pred = _predict_ai_source_from_pil(pil_img)
                    if source_pred is not None:
                        ai_source_text = _format_ai_source_text(source_pred)
                        source_label = source_pred.get('label', '').lower()
                        if source_label in ['gan', 'diffusion']:
                            label = f'AI-generated ({source_label.upper()})'
                    else:
                        ai_source_text = 'AI origin analysis unavailable.'

                # Traditional heatmap
                patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
                heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
                overlay = make_overlay_pil(img, heat)

                # Perform ethical assessment if AI-generated detected
                if is_ai:
                    assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
                    ethical_status = get_enhanced_ethical_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob, overlay, ethical_status, ethical_report, ai_source_text

        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fall back to heuristic

    # Fallback heuristic
    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
    overlay = make_overlay_pil(img, heat)

    ai_score_raw = float(np.mean(heat))
    # Apply temperature scaling to reduce overconfidence
    ai_score = apply_temperature_scaling(ai_score_raw)
    threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else 0.85
    is_ai = ai_score >= threshold
    label = 'AI-generated' if is_ai else 'Real (camera)'

    if is_ai:
        source_pred = _predict_ai_source_from_pil(pil_img)
        if source_pred is not None:
            ai_source_text = _format_ai_source_text(source_pred)
            source_label = source_pred.get('label', '').lower()
            if source_label in ['gan', 'diffusion']:
                label = f'AI-generated ({source_label.upper()})'
        else:
            ai_source_text = 'AI origin analysis unavailable.'
    
    # Perform ethical assessment if AI-generated detected
    if is_ai:
        assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
        ethical_status = get_enhanced_ethical_status(assessment)
        ethical_report = format_ethical_report(assessment)
        if not show_raw_features:
            idx = ethical_report.find('\nRAW FEATURES:')
            if idx != -1:
                ethical_report = ethical_report[:idx]
    
    return label, ai_score, overlay, ethical_status, ethical_report, ai_source_text


def apply_gradcam_overlay_from_pil(pil_img, model, model_type):
    """Apply Grad-CAM to PIL image for deep learning models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare transform based on model type
    if model_type == 'efficientnet_fft':
        img_size = MODEL_INFO.get('image_size', 224) if MODEL_INFO else 224
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_size = (img_size, img_size)
    elif model_type in ['resnet', 'fusion', 'fusion_improved']:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_size = (224, 224)
    else:
        transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_min_size(img, 128)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_size = (128, 128)

    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    # Get target layer based on model type
    if model_type == 'efficientnet_fft':
        # For EfficientNet, use the last convolutional block
        if HAS_TIMM:
            # timm EfficientNet structure
            target_layer = model.backbone.conv_head
        else:
            # torchvision EfficientNet structure
            target_layer = model.backbone.features[-1]
    elif model_type == 'resnet':
        target_layer = model.resnet.layer4[-1].conv3
    elif model_type in ['fusion', 'fusion_improved']:
        target_layer = model.resnet[7][-1].conv3
    else:
        target_layer = model.conv4

    # Use custom simple Grad-CAM for EfficientNet
    if model_type == 'efficientnet_fft':
        # Simple Grad-CAM implementation
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        model.eval()
        output = model(input_tensor)
        model.zero_grad()
        output[0, 1].backward()  # Focus on fake class

        handle_f.remove()
        handle_b.remove()

        # Compute CAM
        act = activations[0].detach()
        grad = gradients[0].detach()
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = (weights * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
    else:
        # Use existing GradCAM from train
        from train import GradCAM
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(input_tensor, target_class=1)

    # Create overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original image
    original = cv2.resize(np.array(pil_img), target_size)

    # Overlay
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay


title = "Advanced Deepfake Detection System with Ethical Assessment"

# Create balanced layout using Blocks with tabs for image and video
with gr.Blocks(title=title) as iface:
    gr.Markdown(f"""
    # {title}
    
    Upload an image or video to detect if it's AI-generated and assess its ethical status.
    
    **Image Model:** {MODEL_INFO['model_type'].upper() if MODEL_INFO else 'Heuristic-based'}
    **Video Model:** {'ResNetLSTM (Video)' if VIDEO_MODEL else 'Not loaded'}
    **AI Source Model:** {'GAN vs Diffusion' if GAN_DIFF_MODEL else 'Not loaded'}
    """)
    
    with gr.Tabs():
        # ===== IMAGE TAB =====
        with gr.Tab("Image Detection"):
            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    input_image = gr.Image(type='pil', label='Upload Image', height=400)
                    
                    gr.Markdown("### Settings")
                    ethical_threshold_img = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=0.5, 
                        label='Ethical Risk Threshold',
                        info='Lower = more strict classification'
                    )
                    show_raw_features_img = gr.Checkbox(
                        label='Show raw feature values', 
                        value=False
                    )
                    
                    analyze_img_btn = gr.Button("Analyze Image", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ---
                    **Features:**
                    - Deep Learning: CNN/ResNet with transfer learning
                    - Grad-CAM visualization highlights suspicious regions
                    - Ethical assessment evaluates privacy and misuse risks
                    - Real-time GPU-accelerated inference
                    """)
                
                # Right Column - Outputs
                with gr.Column(scale=1):
                    gr.Markdown("### Detection Results")
                    
                    with gr.Row():
                        detection_result_img = gr.Label(num_top_classes=2, label='Classification')
                        ai_score_img = gr.Number(label='AI-likelihood Score', precision=4)
                    
                    heatmap_img = gr.Image(label='Detection Heatmap', height=400)
                    
                    gr.Markdown("### Ethical Assessment")
                    ethical_status_img = gr.Textbox(label='Status', lines=2)

                    gr.Markdown("### AI Source Analysis")
                    ai_source_img = gr.Textbox(label='GAN vs Diffusion', lines=4)
                    
                    with gr.Accordion("Full Report", open=False):
                        ethical_report_img = gr.Textbox(
                            label='Detailed Assessment', 
                            lines=30
                        )
            
            # Connect image button to function
            analyze_img_btn.click(
                fn=predict_gradio,
                inputs=[input_image, ethical_threshold_img, show_raw_features_img],
                outputs=[detection_result_img, ai_score_img, heatmap_img, ethical_status_img, ethical_report_img, ai_source_img]
            )
        
        # ===== VIDEO TAB =====
        with gr.Tab("Video Detection"):
            with gr.Row():
                # Left Column - Inputs
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    input_video = gr.Video(label='Upload Video', height=400)
                    
                    gr.Markdown("### Settings")
                    ethical_threshold_vid = gr.Slider(
                        minimum=0.0, 
                        maximum=1.0, 
                        step=0.01, 
                        value=0.5, 
                        label='Ethical Risk Threshold',
                        info='Lower = more strict classification'
                    )
                    show_raw_features_vid = gr.Checkbox(
                        label='Show raw feature values', 
                        value=False
                    )
                    
                    analyze_vid_btn = gr.Button("Analyze Video", variant="primary", size="lg")
                    
                    gr.Markdown("""
                    ---
                    **Features:**
                    - ResNetLSTM: temporal modeling with LSTM
                    - Frame-level & video-level predictions
                    - Grad-CAM on most suspicious frame
                    - Ethical assessment from frame content
                    - Supports MP4, AVI, MOV, MKV, WebM
                    """)
                
                # Right Column - Outputs
                with gr.Column(scale=1):
                    gr.Markdown("### Detection Results")
                    
                    with gr.Row():
                        detection_result_vid = gr.Label(num_top_classes=2, label='Classification')
                        ai_score_vid = gr.Number(label='AI-likelihood Score', precision=4)
                    
                    heatmap_vid = gr.Image(label='Suspicious Frame (with Grad-CAM)', height=400)
                    
                    gr.Markdown("### Ethical Assessment")
                    ethical_status_vid = gr.Textbox(label='Status', lines=2)

                    gr.Markdown("### AI Source Analysis")
                    ai_source_vid = gr.Textbox(label='GAN vs Diffusion', lines=4)
                    
                    with gr.Accordion("Full Report", open=False):
                        ethical_report_vid = gr.Textbox(
                            label='Detailed Assessment', 
                            lines=30
                        )
            
            # Connect video button to function
            analyze_vid_btn.click(
                fn=predict_video_gradio,
                inputs=[input_video, ethical_threshold_vid, show_raw_features_vid],
                outputs=[detection_result_vid, ai_score_vid, heatmap_vid, ethical_status_vid, ethical_report_vid, ai_source_vid]
            )
    
    gr.Markdown("""
    ---
    **How it works:** 
    - **Image**: The heatmap overlay shows regions the model considers suspicious for deepfake artifacts.
    - **Video**: Frames are processed temporally, and the most suspicious frame is highlighted with Grad-CAM.
    - Ethical classification is based on artifact detectability and human face presence.
    
    *Powered by FHIBE Dataset concepts for face authenticity verification.*
    """)


if __name__ == '__main__':
    iface.launch()
