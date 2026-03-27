#!/usr/bin/env python3
"""
Quick Validation Evaluation (Subset)
Evaluates first 2000 samples for fast metrics preview
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve, average_precision_score,
    matthews_corrcoef
)
from torchvision import transforms, models
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Try to import timm for EfficientNet
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class FFTFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.fft_processor = nn.Sequential(
            nn.Linear(12, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Linear(128, output_dim),
        )

    @torch.no_grad()
    def _extract_fft_features(self, x):
        B, C, H, W = x.shape
        x_f32 = x.float()
        gray = 0.299 * x_f32[:, 0] + 0.587 * x_f32[:, 1] + 0.114 * x_f32[:, 2] if C == 3 else x_f32[:, 0]
        fft_img = torch.fft.fft2(gray)
        fft_shift = torch.fft.fftshift(fft_img)
        mag = torch.abs(fft_shift) + 1e-8
        mag = mag / (mag.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] + 1e-8)
        fft_features = []
        for i in range(B):
            m = mag[i].flatten()
            feat = torch.stack([m.mean(), m.std().clamp(min=1e-8), m.max(), m.min(),
                (m > m.mean()).float().mean(), m.median(),
                mag[i][:H//4, :].mean(), mag[i][H//4:H//2, :].mean(),
                mag[i][H//2:3*H//4, :].mean(), mag[i][3*H//4:, :].mean(),
                (m > 0.5).float().mean(), (m > 0.1).float().mean()])
            fft_features.append(torch.clamp(feat, min=-10, max=10))
        return torch.stack(fft_features, dim=0)

    def forward(self, x):
        fft_feat = self._extract_fft_features(x)
        fft_feat = fft_feat.to(x.dtype).detach()
        fft_feat.requires_grad_(True)
        return self.fft_processor(fft_feat)


class EfficientNetFFTFusion(nn.Module):
    def __init__(self, num_classes=2, dropout=0.4, backbone='efficientnet_b2'):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
            backbone_dim = self.backbone.num_features
        else:
            if 'b2' in backbone:
                weights = models.EfficientNet_B2_Weights.IMAGENET1K_V1
                self.backbone = models.efficientnet_b2(weights=weights)
            else:
                weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
                self.backbone = models.efficientnet_b4(weights=weights)
            backbone_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        fft_dim = 512
        self.fft_extractor = FFTFeatureExtractor(output_dim=fft_dim)
        fusion_dim = backbone_dim + fft_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout * 0.5),
        )
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        backbone_feat = self.backbone(x)
        fft_feat = self.fft_extractor(x)
        fused = torch.cat([backbone_feat, fft_feat], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


class ValidationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, balanced=False, max_per_class=None):
        if balanced and max_per_class:
            # Create balanced sample: max_per_class Real + max_per_class Fake
            real_mask = labels == 0
            fake_mask = labels == 1
            real_indices = np.where(real_mask)[0][:max_per_class]
            fake_indices = np.where(fake_mask)[0][:max_per_class]
            indices = np.concatenate([real_indices, fake_indices])
            self.image_paths = image_paths[indices]
            self.labels = labels[indices]
        else:
            self.image_paths = image_paths[:2000] if max_per_class is None else image_paths[:max_per_class]
            self.labels = labels[:2000] if max_per_class is None else labels[:max_per_class]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            return self.transform(img) if self.transform else img, self.labels[idx]
        except:
            return torch.randn(3, 224, 224) * 0.1, self.labels[idx]


def find_class_dir(base_dir, class_names):
    base = Path(base_dir)
    if not base.exists():
        return None
    for name in class_names:
        candidate = base / name
        if candidate.exists():
            return candidate
    for item in base.iterdir():
        if item.is_dir() and item.name.lower() in [n.lower() for n in class_names]:
            return item
    return None


def load_validation_dataset(data_dir='Deepfake vs Real'):
    data_path = Path(data_dir)
    val_dir = find_class_dir(data_path, ['Validation', 'validation', 'Val', 'val'])
    if val_dir is None:
        return np.array([]), np.array([])

    image_paths, labels = [], []
    real_dir = find_class_dir(val_dir, ['Real', 'REAL', 'real'])
    if real_dir and real_dir.exists():
        for img_path in list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')) + list(real_dir.glob('*.jpeg')):
            image_paths.append(img_path)
            labels.append(0)

    fake_dir = find_class_dir(val_dir, ['Fake', 'FAKE', 'fake'])
    if fake_dir and fake_dir.exists():
        for img_path in list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')) + list(fake_dir.glob('*.jpeg')):
            image_paths.append(img_path)
            labels.append(1)

    return np.array(image_paths), np.array(labels)


def compute_metrics(y_true, y_pred, y_probs):
    m = {}
    m['accuracy'] = accuracy_score(y_true, y_pred)
    m['precision_fake'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    m['precision_real'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    m['recall_fake'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    m['recall_real'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    m['f1_fake'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    m['f1_real'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    m['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    m['auc_roc'] = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
    m['auc_pr'] = average_precision_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5

    cm = confusion_matrix(y_true, y_pred)
    m['cm'] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    m['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    m['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    m['mcc'] = matthews_corrcoef(y_true, y_pred)
    return m


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n" + "=" * 90)
    print(" QUICK VALIDATION EVALUATION (2000 SAMPLES - BALANCED)")
    print("=" * 90 + f"\nDevice: {device}\n")

    print("Loading validation dataset...")
    val_img, val_lbl = load_validation_dataset('Deepfake vs Real')
    print(f"Total loaded: {len(val_img)} (Real: {(val_lbl == 0).sum()}, Fake: {(val_lbl == 1).sum()})\n")

    print("Loading model...")
    model_path = Path('models_adv/best_model_weights.pt')
    if not model_path.exists():
        print(f"ERROR: {model_path} not found!")
        return

    config_path = Path('models_adv/config.json')
    config = json.load(open(config_path)) if config_path.exists() else {}

    model = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone='efficientnet_b2')
    state_dict = torch.load(model_path, map_location=device)

    # Auto-detect backbone
    fusion_in_dim = state_dict['fusion.0.weight'].shape[1]
    detected_backbone = {1280: 'efficientnet_b0', 1408: 'efficientnet_b2', 1792: 'efficientnet_b4'}.get(fusion_in_dim - 512, 'efficientnet_b2')
    print(f"Detected backbone: {detected_backbone}")

    model = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone=detected_backbone)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print("Model loaded!\n")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ValidationDataset(val_img, val_lbl, transform, balanced=True, max_per_class=1000)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    print(f"Running inference on {len(dataset)} samples (1000 Real + 1000 Fake)...\n")
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("Computing metrics...\n")
    m = compute_metrics(all_labels, all_preds, all_probs)

    # Print report
    print("=" * 90)
    print(" METRICS REPORT (2000 BALANCED SAMPLES)")
    print("=" * 90)
    print(f"\n[OVERVIEW]")
    print(f"  Accuracy:           {m['accuracy']*100:.2f}%")
    print(f"  AUC-ROC:            {m['auc_roc']:.4f}")
    print(f"  AUC-PR:             {m['auc_pr']:.4f}")
    print(f"  MCC:                {m['mcc']:.4f}")

    print(f"\n[FAKE DETECTION]")
    print(f"  Precision:          {m['precision_fake']*100:.2f}%")
    print(f"  Recall/Sensitivity: {m['recall_fake']*100:.2f}%")
    print(f"  F1-Score:           {m['f1_fake']:.4f}")

    print(f"\n[REAL DETECTION]")
    print(f"  Precision:          {m['precision_real']*100:.2f}%")
    print(f"  Recall/Specificity: {m['recall_real']*100:.2f}%")
    print(f"  F1-Score:           {m['f1_real']:.4f}")

    print(f"\n[ROC ANALYSIS]")
    print(f"  Sensitivity (TPR):  {m['sensitivity']*100:.2f}%")
    print(f"  Specificity (TNR):  {m['specificity']*100:.2f}%")

    print(f"\n[CONFUSION MATRIX]")
    cm = np.array(m['cm'])
    print(f"              Predicted")
    print(f"              Real    Fake")
    print(f"  Actual Real  {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Actual Fake  {cm[1,0]:5d}   {cm[1,1]:5d}")

    print(f"\n[CLASS DISTRIBUTION]")
    print(f"  Real samples: {(all_labels == 0).sum()}")
    print(f"  Fake samples: {(all_labels == 1).sum()}")

    print("=" * 90 + "\n")

    # Save report
    report = {
        'timestamp': datetime.now().isoformat(),
        'type': 'Quick validation (2000 samples)',
        'model': detected_backbone,
        'metrics': {k: float(v) if isinstance(v, (int, np.integer, np.floating)) else v
                    for k, v in m.items()}
    }
    with open('models_adv/validation_metrics_quick.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Report saved to: models_adv/validation_metrics_quick.json\n")


if __name__ == '__main__':
    main()
