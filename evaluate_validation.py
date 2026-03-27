#!/usr/bin/env python3
"""
Comprehensive Validation Set Evaluation
Computes all metrics: Accuracy, AUC-ROC, Precision, Recall, F1, Confusion Matrix, etc.
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
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, matthews_corrcoef
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
    print("Warning: timm not installed. Install with: pip install timm")


# ============================================================================
# FFT FEATURE EXTRACTOR
# ============================================================================
class FFTFeatureExtractor(nn.Module):
    """Extract and process FFT features"""
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
                mag[i][:H//4, :].mean(),
                mag[i][H//4:H//2, :].mean(),
                mag[i][H//2:3*H//4, :].mean(),
                mag[i][3*H//4:, :].mean(),
                (m > 0.5).float().mean(),
                (m > 0.1).float().mean(),
            ])
            feat = torch.clamp(feat, min=-10, max=10)
            fft_features.append(feat)

        return torch.stack(fft_features, dim=0)

    def forward(self, x):
        fft_feat = self._extract_fft_features(x)
        fft_feat = fft_feat.to(x.dtype).detach()
        fft_feat.requires_grad_(True)
        return self.fft_processor(fft_feat)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
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


# ============================================================================
# DATASET
# ============================================================================
class ValidationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx], str(self.image_paths[idx])
        except Exception as e:
            noise = torch.randn(3, 224, 224) * 0.1
            return noise, self.labels[idx], "ERROR"


# ============================================================================
# UTILITIES
# ============================================================================
def find_class_dir(base_dir, class_names):
    """Find directory matching any of the class names"""
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


def load_validation_dataset(data_dir='DeepfakeVsReal/Dataset', max_per_class=None):
    """Load validation/test dataset"""
    data_path = Path(data_dir)

    # Try to find validation or test directory
    val_dir = find_class_dir(data_path, ['Validation', 'validation', 'Val', 'val'])
    if val_dir is None:
        val_dir = find_class_dir(data_path, ['Test', 'test', 'testing'])
    if val_dir is None:
        print(f"No Validation or Test directory found in {data_path}")
        return np.array([]), np.array([])

    image_paths = []
    labels = []

    real_dir = find_class_dir(val_dir, ['Real', 'REAL', 'real'])
    if real_dir and real_dir.exists():
        real_images = list(real_dir.glob('*.jpg')) + list(real_dir.glob('*.png')) + list(real_dir.glob('*.jpeg'))
        if max_per_class:
            real_images = real_images[:max_per_class]
        for img_path in real_images:
            image_paths.append(img_path)
            labels.append(0)
        print(f"  Loaded {len(real_images)} REAL images")

    fake_dir = find_class_dir(val_dir, ['Fake', 'FAKE', 'fake'])
    if fake_dir and fake_dir.exists():
        fake_images = list(fake_dir.glob('*.jpg')) + list(fake_dir.glob('*.png')) + list(fake_dir.glob('*.jpeg'))
        if max_per_class:
            fake_images = fake_images[:max_per_class]
        for img_path in fake_images:
            image_paths.append(img_path)
            labels.append(1)
        print(f"  Loaded {len(fake_images)} FAKE images")

    return np.array(image_paths), np.array(labels)


def compute_all_metrics(y_true, y_pred, y_probs):
    """Compute comprehensive metrics"""
    metrics = {}

    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision_fake'] = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['precision_real'] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['recall_fake'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['recall_real'] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1_fake'] = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    metrics['f1_real'] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # AUC metrics
    try:
        metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
    except:
        metrics['auc_roc'] = 0.5

    try:
        metrics['auc_pr'] = average_precision_score(y_true, y_probs)
    except:
        metrics['auc_pr'] = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    tn, fp, fn, tp = cm.ravel()
    metrics['tn'] = int(tn)
    metrics['fp'] = int(fp)
    metrics['fn'] = int(fn)
    metrics['tp'] = int(tp)

    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    metrics['mcc'] = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0

    return metrics


def print_metrics_report(metrics):
    """Print detailed metrics report"""
    print("\n" + "=" * 90)
    print(" COMPREHENSIVE VALIDATION METRICS REPORT")
    print("=" * 90)

    # Overview
    print("\n[OVERVIEW]")
    print(f"  Accuracy:           {metrics['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']*100:.2f}%")
    print(f"  AUC-ROC:            {metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:             {metrics['auc_pr']:.4f}")
    print(f"  MCC:                {metrics['mcc']:.4f}")

    # Fake Detection
    print("\n[FAKE DETECTION METRICS]")
    print(f"  Precision:          {metrics['precision_fake']*100:.2f}%")
    print(f"  Recall (Sensitivity): {metrics['recall_fake']*100:.2f}%")
    print(f"  F1-Score:           {metrics['f1_fake']:.4f}")

    # Real Detection
    print("\n[REAL DETECTION METRICS]")
    print(f"  Precision:          {metrics['precision_real']*100:.2f}%")
    print(f"  Recall (Specificity): {metrics['recall_real']*100:.2f}%")
    print(f"  F1-Score:           {metrics['f1_real']:.4f}")

    # ROC/Performance
    print("\n[ROC ANALYSIS]")
    print(f"  True Positive Rate (Sensitivity):  {metrics['sensitivity']*100:.2f}%")
    print(f"  True Negative Rate (Specificity):  {metrics['specificity']*100:.2f}%")
    print(f"  False Positive Rate:               {metrics['fpr']*100:.2f}%")
    print(f"  False Negative Rate:               {metrics['fnr']*100:.2f}%")

    # Confusion Matrix
    print("\n[CONFUSION MATRIX]")
    cm = np.array(metrics['confusion_matrix'])
    print(f"                Predicted")
    print(f"                Real    Fake")
    print(f"  Actual Real   {cm[0,0]:5d}   {cm[0,1]:5d}")
    print(f"  Actual Fake   {cm[1,0]:5d}   {cm[1,1]:5d}")

    print("\n[CONFUSION MATRIX BREAKDOWN]")
    print(f"  True Negatives (TN):   {metrics['tn']}")
    print(f"  False Positives (FP):  {metrics['fp']}")
    print(f"  False Negatives (FN):  {metrics['fn']}")
    print(f"  True Positives (TP):   {metrics['tp']}")

    print("\n" + "=" * 90 + "\n")


# ============================================================================
# MAIN EVALUATION
# ============================================================================
def main():
    print("\n" + "=" * 90)
    print(" VALIDATION SET EVALUATION")
    print("=" * 90)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load validation data
    print("Loading validation dataset...")
    val_img, val_lbl = load_validation_dataset('Deepfake vs Real')

    if len(val_img) == 0:
        print("ERROR: No validation images found!")
        return

    print(f"Total validation samples: {len(val_img)}")
    print(f"  Real: {(val_lbl == 0).sum()}")
    print(f"  Fake: {(val_lbl == 1).sum()}\n")

    # Load model
    print("Loading model from models_adv/best_model_weights.pt...")
    model_path = Path('models_adv/best_model_weights.pt')
    config_path = Path('models_adv/config.json')

    if not model_path.exists():
        print(f"ERROR: Model file not found at {model_path}")
        return

    # Load config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    backbone = config.get('backbone', 'efficientnet_b2')
    print(f"Backbone: {backbone}")

    # Create and load model
    model = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone='efficientnet_b2')
    state_dict = torch.load(model_path, map_location=device)

    # Auto-detect backbone from fusion layer input dimension
    fusion_in_dim = state_dict['fusion.0.weight'].shape[1]
    backbone_dim = fusion_in_dim - 512
    backbone_map = {1280: 'efficientnet_b0', 1408: 'efficientnet_b2', 1792: 'efficientnet_b4'}
    detected_backbone = backbone_map.get(backbone_dim, 'efficientnet_b2')
    print(f"Detected backbone: {detected_backbone} (dim={backbone_dim})\n")

    # Recreate model with correct backbone
    model = EfficientNetFFTFusion(num_classes=2, dropout=0.4, backbone=detected_backbone)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully!\n")

    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = ValidationDataset(val_img, val_lbl, transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Run inference
    print("Running inference on validation set...")
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels, paths in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f"Processed {len(all_labels)} samples\n")

    # Compute metrics
    print("Computing metrics...")
    metrics = compute_all_metrics(all_labels, all_preds, all_probs)

    # Print report
    print_metrics_report(metrics)

    # Save report to JSON
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'backbone': backbone,
            'model_path': str(model_path),
        },
        'dataset_info': {
            'total_samples': len(all_labels),
            'real_samples': int((all_labels == 0).sum()),
            'fake_samples': int((all_labels == 1).sum()),
        },
        'metrics': {k: v if not isinstance(v, np.ndarray) else v.tolist()
                    for k, v in metrics.items()},
    }

    output_path = 'models_adv/validation_metrics.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {output_path}\n")

    # Summary
    print("=" * 90)
    print(" SUMMARY")
    print("=" * 90)
    print(f"Accuracy:      {metrics['accuracy']*100:.2f}%")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"F1 (Macro):    {metrics['f1_macro']:.4f}")
    print(f"Sensitivity:   {metrics['sensitivity']*100:.2f}% (detect fakes)")
    print(f"Specificity:   {metrics['specificity']*100:.2f}% (detect real)")
    print("=" * 90 + "\n")


if __name__ == '__main__':
    main()
