import argparse
import os
import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import expon, randint

# PyTorch imports for CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import cv2
import io
import time
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from detector import load_image, rgb_to_gray, extract_residual, fft_stats, lbp_entropy


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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Standard cross-entropy with label smoothing
        ce_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(inputs, targets)
        pt = torch.exp(-ce_loss)
        
        # Focal loss formula: -α(1-pt)^γ * log(pt)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss


class MixupDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, mixup_alpha=0.2):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.mixup_alpha = mixup_alpha

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load primary image
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Mixup with probability 0.5
        if np.random.random() > 0.5:
            # Select random second image
            mix_idx = np.random.randint(0, len(self.file_paths))
            mix_img_path = self.file_paths[mix_idx]
            mix_image = Image.open(mix_img_path).convert('RGB')
            mix_label = self.labels[mix_idx]

            if self.transform:
                mix_image = self.transform(mix_image)

            # Mixup lambda
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            image = lam * image + (1 - lam) * mix_image
            
            # Mixed labels for loss computation
            return image, (label, mix_label, lam)
        
        return image, (label, label, 1.0)  # No mixup


class DeepfakeCNN(nn.Module):
    def __init__(self):
        super(DeepfakeCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = self.global_pool(x)
        x = x.view(-1, 512)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class DeepfakeResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DeepfakeResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        # Replace final layer for binary classification
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 2)
        )

    def forward(self, x):
        return self.resnet(x)


class DeepfakeFeatureFusion(nn.Module):
    def __init__(self, resnet_pretrained=True, block_size=16, attn_dim=512, attn_heads=8):
        super(DeepfakeFeatureFusion, self).__init__()
        self.block_size = block_size

        # ResNet backbone for spatial features
        self.resnet = models.resnet50(pretrained=resnet_pretrained)
        # Remove the final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Enhanced frequency feature extractor with more layers
        self.freq_conv = nn.Sequential(
            nn.Conv2d(6, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Cross-attention to align spatial cues with frequency anomalies
        self.spatial_proj = nn.Linear(2048, attn_dim)
        self.freq_proj = nn.Linear(256, attn_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True)

        # Feature fusion and classification with adaptive dropout
        # ResNet features: 2048, Frequency features: 256, Cross-attn: attn_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(2048 + 256 + attn_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def extract_frequency_features(self, x):
        """Extract block-wise frequency features using FFT magnitude + phase."""
        b, c, h, w = x.shape
        bs = self.block_size

        pad_h = (bs - (h % bs)) % bs
        pad_w = (bs - (w % bs)) % bs
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Split into non-overlapping blocks: B x C x Gh x Gw x bs x bs
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)
        gh, gw = blocks.shape[2], blocks.shape[3]
        blocks = blocks.contiguous().view(b, c, gh * gw, bs, bs)

        # FFT per block
        x_fft = torch.fft.fft2(blocks, dim=(-2, -1))
        magnitude = torch.log1p(torch.abs(x_fft))
        phase = torch.angle(x_fft)

        # Normalize per block
        magnitude = (magnitude - magnitude.mean(dim=(-2, -1), keepdim=True)) / (
            magnitude.std(dim=(-2, -1), keepdim=True) + 1e-8
        )
        phase = (phase - phase.mean(dim=(-2, -1), keepdim=True)) / (
            phase.std(dim=(-2, -1), keepdim=True) + 1e-8
        )

        # Aggregate block statistics into a grid
        mag_mean = magnitude.mean(dim=(-2, -1))
        phase_mean = phase.mean(dim=(-2, -1))
        freq_grid = torch.cat([mag_mean, phase_mean], dim=1)
        freq_grid = freq_grid.view(b, 2 * c, gh, gw)

        freq_out = self.freq_conv(freq_grid)
        return freq_out.view(freq_out.size(0), -1)

    def forward(self, x):
        # Spatial features from ResNet
        spatial_features = self.resnet(x)
        spatial_features = spatial_features.view(spatial_features.size(0), -1)

        # Frequency features
        freq_features = self.extract_frequency_features(x)

        # Cross-attention: spatial queries, frequency keys/values
        q = self.spatial_proj(spatial_features).unsqueeze(1)
        kv = self.freq_proj(freq_features).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        attn_out = attn_out.squeeze(1)

        # Concatenate features
        fused_features = torch.cat([spatial_features, freq_features, attn_out], dim=1)

        # Classification
        output = self.fusion_classifier(fused_features)
        return output


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to capture gradients and activations
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)

        # Get the score for the target class
        target = output[:, target_class]

        # Backward pass to get gradients
        self.model.zero_grad()
        target.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Calculate weights
        weights = np.mean(gradients, axis=(1, 2))

        # Generate CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)

        return cam


def apply_gradcam_overlay(image_path, model, target_class=1, alpha=0.5):
    """Apply Grad-CAM visualization to an image and return the overlay"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Get target layer (last conv layer of ResNet)
    target_layer = model.resnet.layer4[-1].conv3

    # Generate CAM
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor, target_class)

    # Convert to heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original image
    original_image = cv2.resize(np.array(image), (224, 224))

    # Overlay heatmap on original image
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)

    return overlay, cam


class RandomJPEGCompression(object):
    """PIL-based random JPEG compression transform to improve JPEG robustness."""
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if np.random.random() > self.p:
            return img
        quality = int(np.random.randint(self.quality_range[0], self.quality_range[1] + 1))
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        return compressed


def compute_class_weights(labels, device):
    """Compute inverse-frequency class weights for balanced loss."""
    counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _tta_flip_variations(tensor_batch):
    """Return list of tensor batches with basic TTA flips applied."""
    # tensor_batch shape: (B,C,H,W)
    variants = []
    variants.append(tensor_batch)  # identity
    variants.append(torch.flip(tensor_batch, dims=[3]))  # hflip
    variants.append(torch.flip(tensor_batch, dims=[2]))  # vflip
    variants.append(torch.flip(tensor_batch, dims=[2, 3]))  # both
    return variants


def tta_predict_batch(model, inputs, device, tta_times=4):
    """Apply simple flip-based TTA and average softmax probabilities."""
    model.eval()
    variants = _tta_flip_variations(inputs)
    probs_accum = None
    with torch.no_grad():
        for v in variants[:tta_times]:
            v = v.to(device)
            out = model(v)
            prob = torch.softmax(out, dim=1)[:, 1].detach().cpu().numpy()
            if probs_accum is None:
                probs_accum = prob
            else:
                probs_accum += prob
    probs_accum = probs_accum / min(len(variants), tta_times)
    return probs_accum


class DeepfakeDualStream(nn.Module):
    """Dual-stream fusion: spatial ResNet + residual high-pass stream."""
    def __init__(self, resnet_pretrained=True):
        super(DeepfakeDualStream, self).__init__()
        self.resnet = models.resnet50(pretrained=resnet_pretrained)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Residual / high-pass stream
        self.res_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )

    def high_pass(self, x):
        # simple Laplacian-like high-pass using conv kernel
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        kernel = kernel.repeat(x.size(1), 1, 1, 1)
        # pad to keep same size
        hp = nn.functional.conv2d(x, kernel, padding=1, groups=x.size(1))
        return hp

    def forward(self, x):
        spatial = self.resnet(x)
        spatial = spatial.view(spatial.size(0), -1)

        residual = self.high_pass(x)
        rfeat = self.res_conv(residual)
        rfeat = rfeat.view(rfeat.size(0), -1)

        fused = torch.cat([spatial, rfeat], dim=1)
        out = self.classifier(fused)
        return out


def train_stacked_classifier(X, y, n_splits=5, random_state=42):
    """Train a stacked meta-classifier (RF + XGB + HGB base learners, LR meta)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_clfs = [
        RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=random_state),
        XGBClassifier(tree_method='hist', device='cpu', n_estimators=200, random_state=random_state),
        HistGradientBoostingClassifier(random_state=random_state)
    ]

    # OOF predictions
    meta_features = np.zeros((X.shape[0], len(base_clfs)))

    for i, clf in enumerate(base_clfs):
        oof = np.zeros(X.shape[0])
        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]
            clf.fit(X_tr, y_tr)
            oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        meta_features[:, i] = oof

    # Train meta classifier
    meta = LogisticRegression(max_iter=1000)
    meta.fit(meta_features, y)

    # Fit base learners on full data for inference
    for clf in base_clfs:
        clf.fit(X, y)

    stacked = {
        'base_learners': base_clfs,
        'meta_learner': meta
    }
    return stacked


class ImageDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def main_cnn(dataset_root, out_model, max_per_class=500, max_val_images=None):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'

    print('Using Enhanced CNN for deepfake detection')
    print('Features: Extended epoch support (up to 100), improved regularization, AI image detection')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect image paths
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)
    print(f'Training on {len(all_files)} images...')

    # Create validation dataloader for early stopping
    if val_root.exists():
        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    else:
        val_dataloader = None
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandomJPEGCompression(quality_range=(50,95), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(all_files, all_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Initialize model with MORE dropout
    model = DeepfakeCNN().to(device)
    # Add L2 regularization
    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)  # Increased weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Training loop with validation-based early stopping
    model.train()
    best_val_acc = 0.0
    best_f1 = 0.0
    best_val_loss = float('inf')
    min_delta = 1e-3
    min_delta_acc = 0.1
    patience = 10  # Extended patience
    patience_counter = 0
    max_epochs = 100  # Support for extended training

    for epoch in range(max_epochs):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        # Validation phase every epoch
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    # loss computed from single forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # TTA-based probabilities for more robust metric estimation
                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_total += labels.size(0)
                    val_correct += (preds == labels.cpu().numpy()).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_dataloader)

            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

            # Early stopping based on val loss and accuracy
            improved = (val_loss < best_val_loss - min_delta) or (val_acc > best_val_acc + min_delta_acc)
            if improved:
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), out_model + '_best')
                print(f'✓ Best model saved')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    # Load best model
                    model.load_state_dict(torch.load(out_model + '_best'))
                    break
            scheduler.step(val_loss)
        else:
            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            scheduler.step(epoch_loss)

        model.train()

    # validation
    if val_root.exists():
        print('Evaluating on Validation set...')
        model.eval()

        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print('Val accuracy:', accuracy_score(all_labels, all_preds))
        try:
            print('Val ROC AUC:', roc_auc_score(all_labels, all_probs))
        except Exception:
            pass
        print(classification_report(all_labels, all_preds))

    # save model
    torch.save(model.state_dict(), out_model)
    # Also save model architecture info
    model_info = {
        'model_type': 'cnn',
        'state_dict_path': out_model,
        'class': 'DeepfakeCNN'
    }
    joblib.dump(model_info, out_model + '_info.pkl')
    print(f'Saved CNN model to {out_model}')
    print(f'Model trained with extended epoch support for better convergence')


def main_cnn_kfold(dataset_root, out_model, max_per_class=500, max_val_images=None, k_folds=5):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'

    print(f'Using {k_folds}-fold cross-validation for CNN training.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect all training files
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f'Total training samples: {len(all_files)}')

    # K-fold cross-validation
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)

    fold_results = []
    best_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
        print(f'\n--- Fold {fold+1}/{k_folds} ---')

        # Split data for this fold
        train_files = [all_files[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        val_files = [all_files[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]

        # Data transforms
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Create datasets
        train_dataset = ImageDataset(train_files, train_labels, transform=train_transform)
        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        # Initialize model for this fold
        model = DeepfakeCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Train this fold
        best_val_acc = 0.0
        patience = 5
        patience_counter = 0

        for epoch in range(20):  # Fewer epochs per fold
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_total += labels.size(0)
                    val_correct += (preds == labels.cpu().numpy()).sum().item()

            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            train_loss = train_loss / len(train_dataloader)
            val_loss = val_loss / len(val_dataloader)

            print(f'Epoch {epoch+1}/20, Train: {train_acc:.1f}%, Val: {val_acc:.1f}%')

            scheduler.step()

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model for this fold
                torch.save(model.state_dict(), f'{out_model}_fold_{fold+1}_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break

        # Load best model for this fold and evaluate
        model.load_state_dict(torch.load(f'{out_model}_fold_{fold+1}_best'))
        model.eval()

        # Final evaluation on this fold's validation set
        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        fold_acc = accuracy_score(all_labels, all_preds)
        fold_auc = roc_auc_score(all_labels, all_probs)

        fold_results.append({
            'fold': fold+1,
            'accuracy': fold_acc,
            'auc': fold_auc,
            'predictions': all_preds,
            'probabilities': all_probs,
            'true_labels': all_labels
        })

        print(f'Fold {fold+1} Results - Accuracy: {fold_acc:.4f}, AUC: {fold_auc:.4f}')

        best_models.append(model)

    # Aggregate results across folds
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    avg_auc = np.mean([r['auc'] for r in fold_results])
    std_auc = np.std([r['auc'] for r in fold_results])

    print(f'\n=== {k_folds}-Fold Cross-Validation Results ===')
    print(f'Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}')
    print(f'Average AUC: {avg_auc:.4f} ± {std_auc:.4f}')

    # Save ensemble model (average of all fold models)
    ensemble_state_dict = {}
    for key in best_models[0].state_dict().keys():
        ensemble_state_dict[key] = torch.stack([model.state_dict()[key] for model in best_models]).mean(0)

    torch.save(ensemble_state_dict, out_model)

    model_info = {
        'model_type': 'cnn_kfold',
        'state_dict_path': out_model,
        'class': 'DeepfakeCNN',
        'k_folds': k_folds,
        'cv_results': {
            'avg_accuracy': avg_accuracy,
            'std_accuracy': std_accuracy,
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'fold_results': fold_results
        }
    }
    joblib.dump(model_info, out_model + '_info.pkl')

    print(f'Saved {k_folds}-fold ensemble CNN model to {out_model}')
    print(f'Expected accuracy on new data: {avg_accuracy:.1f}% ± {std_accuracy*100:.1f}%')


def main_resnet(dataset_root, out_model, max_per_class=500, max_val_images=None):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'

    print('Using pre-trained ResNet50 with transfer learning.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect image paths
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f'Training on {len(all_files)} images...')

    # Create validation dataloader for early stopping
    if val_root.exists():
        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    else:
        val_dataloader = None

    # Data transforms for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandomJPEGCompression(quality_range=(50,95), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = ImageDataset(all_files, all_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    # Initialize ResNet model
    model = DeepfakeResNet(pretrained=True).to(device)

    # Freeze early layers for transfer learning
    for param in model.resnet.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers
    for param in model.resnet.layer3.parameters():
        param.requires_grad = True
    for param in model.resnet.layer4.parameters():
        param.requires_grad = True
    for param in model.resnet.fc.parameters():
        param.requires_grad = True

    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam([
        {'params': model.resnet.layer3.parameters(), 'lr': 1e-4},
        {'params': model.resnet.layer4.parameters(), 'lr': 1e-4},
        {'params': model.resnet.fc.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    # Training loop
    model.train()
    best_val_acc = 0.0
    patience = 7
    patience_counter = 0
    best_val_loss = float('inf')
    min_delta = 1e-3
    min_delta_acc = 0.1

    for epoch in range(20):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_dataloader)

            print(f'Epoch {epoch+1}/20, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            # Early stopping based on val loss and accuracy
            improved = (val_loss < best_val_loss - min_delta) or (val_acc > best_val_acc + min_delta_acc)
            if improved:
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                patience_counter = 0
                torch.save(model.state_dict(), out_model + '_best')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}, best val acc: {best_val_acc:.2f}%')
                    model.load_state_dict(torch.load(out_model + '_best'))
                    break
            scheduler.step(val_loss)
        else:
            print(f'Epoch {epoch+1}/20, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
            scheduler.step(epoch_loss)

        model.train()

    print('ResNet training complete!')

    # Final evaluation
    if val_root.exists():
        print('Evaluating on Validation set...')
        model.eval()

        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print('Val accuracy:', accuracy_score(all_labels, all_preds))
        try:
            print('Val ROC AUC:', roc_auc_score(all_labels, all_probs))
        except Exception:
            pass
        print(classification_report(all_labels, all_preds))

    # Save model
    torch.save(model.state_dict(), out_model)
    model_info = {
        'model_type': 'resnet',
        'state_dict_path': out_model,
        'class': 'DeepfakeResNet'
    }
    joblib.dump(model_info, out_model + '_info.pkl')
    print('Saved ResNet model to', out_model)


def main_fusion_improved(dataset_root, out_model, max_per_class=500, max_val_images=None):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'

    print('Using ADVANCED ResNet + FFT Feature Fusion with AI Detection')
    print('Features: Extended training (up to 100 epochs), better AI/Synthetic detection, enhanced regularization')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect image paths
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f'Training on {len(all_files)} images...')

    # Create validation dataloader
    if val_root.exists():
        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    else:
        val_dataloader = None

    # Enhanced data transforms optimized for AI image detection
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        RandomJPEGCompression(quality_range=(40,90), p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.RandomErasing(p=0.5, scale=(0.02, 0.1))], p=0.2)  # After ToTensor
    ])

    # Create dataset with Mixup
    dataset = MixupDataset(all_files, all_labels, transform=transform, mixup_alpha=0.2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    # Initialize model
    model = DeepfakeFeatureFusion(resnet_pretrained=True).to(device)

    # Freeze early ResNet layers
    for param in model.resnet.parameters():
        param.requires_grad = False

    # Unfreeze later layers
    for param in model.resnet[6].parameters():  # layer3
        param.requires_grad = True
    for param in model.resnet[7].parameters():  # layer4
        param.requires_grad = True

    # Focal Loss for hard example mining and better AI detection
    criterion = FocalLoss(alpha=0.8, gamma=2.5, label_smoothing=0.15)

    # Optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': model.resnet[6].parameters(), 'lr': 3e-5},  # Lower LR for pretrained layers
        {'params': model.resnet[7].parameters(), 'lr': 3e-5},
        {'params': model.freq_conv.parameters(), 'lr': 5e-4},  # Moderate LR for frequency features
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-3}  # Higher LR for classifier
    ], weight_decay=5e-4, betas=(0.9, 0.999))

    # Cosine annealing with warm restarts for extended training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-7)

    # Training loop with extended epochs
    model.train()
    best_f1_score = 0.0
    best_fake_recall = 0.0
    best_real_recall = 0.0
    patience = 12  # Longer patience for extended training
    patience_counter = 0
    max_epochs = 100  # Support for extended training

    for epoch in range(max_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, label_data in dataloader:
            inputs = inputs.to(device)
            
            # Handle mixup labels
            if len(label_data) == 3:
                labels_a, labels_b, lam = label_data
                labels_a, labels_b = labels_a.to(device), labels_b.to(device)
                lam = lam.to(device).float()
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Mixup loss
                loss_a = criterion(outputs, labels_a)
                loss_b = criterion(outputs, labels_b)
                loss = (lam * loss_a + (1 - lam) * loss_b).mean()
                
                # Use primary labels for accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels_a.size(0)
                correct += (predicted == labels_a).sum().item()
            else:
                labels = label_data[0].to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        scheduler.step()

        # Validation
        if val_dataloader is not None:
            model.eval()
            val_preds = []
            val_true = []
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    # Use TTA for predictions
                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_preds.extend(preds.tolist())
                    val_true.extend(labels.cpu().numpy().tolist())

            val_loss = val_loss / len(val_dataloader)
            val_acc = 100 * accuracy_score(val_true, val_preds)

            # Calculate per-class metrics
            from sklearn.metrics import classification_report, f1_score
            report = classification_report(val_true, val_preds, output_dict=True, zero_division=0)
            fake_recall = report['1']['recall']  # Class 1 (fake) recall
            f1 = f1_score(val_true, val_preds, average='macro')

            real_recall = report['0']['recall']  # Class 0 (real) recall
            print(f'Epoch {epoch+1}/{max_epochs}, Train: {epoch_acc:.1f}%, Val: {val_acc:.1f}%, Real Recall: {real_recall:.3f}, Fake Recall: {fake_recall:.3f}, F1: {f1:.3f}')

            # Save best model based on balanced detection (both real and fake)
            combined_score = (fake_recall + real_recall) / 2  # Balanced metric
            if combined_score > (best_fake_recall + best_real_recall) / 2 or \
               (abs(combined_score - (best_fake_recall + best_real_recall) / 2) < 0.02 and f1 > best_f1_score):
                best_fake_recall = fake_recall
                best_real_recall = real_recall
                best_f1_score = f1
                patience_counter = 0
                torch.save(model.state_dict(), out_model + '_best_improved')
                print(f'✓ Best model saved (Real Recall: {real_recall:.3f}, Fake Recall: {fake_recall:.3f}, F1: {f1:.3f})')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                    model.load_state_dict(torch.load(out_model + '_best_improved'))
                    break
        else:
            print(f'Epoch {epoch+1}/{max_epochs}, Train: {epoch_acc:.1f}%')

        model.train()

    print('Improved feature fusion training complete!')

    # Final evaluation with threshold optimization
    if val_root.exists():
        print('\\nFinal evaluation with threshold optimization...')
        model.eval()

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Find optimal threshold for fake detection
        from sklearn.metrics import precision_recall_curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Apply optimal threshold
        optimized_preds = (np.array(all_probs) >= optimal_threshold).astype(int)

        print(f'Optimal threshold: {optimal_threshold:.3f}')
        print(f'Standard (0.5) accuracy: {accuracy_score(all_labels, all_preds):.4f}')
        print(f'Optimized accuracy: {accuracy_score(all_labels, optimized_preds):.4f}')
        print('\\nOptimized Classification Report:')
        print(classification_report(all_labels, optimized_preds))

        # Save model with threshold info
        model_info = {
            'model_type': 'fusion_improved',
            'state_dict_path': out_model + '_best_improved',
            'class': 'DeepfakeFeatureFusion',
            'optimal_threshold': optimal_threshold,
            'best_fake_recall': best_fake_recall,
            'best_f1_score': best_f1_score
        }
        joblib.dump(model_info, out_model + '_info.pkl')

    # Save final model
    torch.save(model.state_dict(), out_model)
    print(f'Saved improved fusion model to {out_model}')


def main_fusion(dataset_root, out_model, max_per_class=500, max_val_images=None):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'

    print('Using ResNet + FFT Feature Fusion with Extended Training Support')
    print('Features: Up to 75+ epochs, AI image optimizations, balanced real/fake detection')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect image paths
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    print(f'Training on {len(all_files)} images...')

    # Create validation dataloader for early stopping
    if val_root.exists():
        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)  # Smaller batch for fusion
    else:
        val_dataloader = None

    # Data transforms for fusion model
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandomJPEGCompression(quality_range=(50,95), p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset with Mixup augmentation for better generalization
    dataset = MixupDataset(all_files, all_labels, transform=transform, mixup_alpha=0.2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True if device.type == 'cuda' else False)

    # Initialize fusion model
    model = DeepfakeFeatureFusion(resnet_pretrained=True).to(device)

    # Freeze early ResNet layers for transfer learning
    for param in model.resnet.parameters():
        param.requires_grad = False

    # Unfreeze later layers
    for param in model.resnet[6].parameters():  # layer3
        param.requires_grad = True
    for param in model.resnet[7].parameters():  # layer4
        param.requires_grad = True

    # Unfreeze frequency layers
    for param in model.freq_conv.parameters():
        param.requires_grad = True

    # Unfreeze fusion classifier
    for param in model.fusion_classifier.parameters():
        param.requires_grad = True

    # Focal Loss with class weights for better fake detection
    # Class weights: More penalty for missing fakes (class 1)
    class_weights = torch.tensor([0.4, 1.6]).to(device)  # Prioritize fake detection
    criterion = FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.1)

    # Different learning rates for different parts
    optimizer = optim.AdamW([  # AdamW with weight decay
        {'params': model.resnet[6].parameters(), 'lr': 5e-5},  # layer3
        {'params': model.resnet[7].parameters(), 'lr': 5e-5},  # layer4
        {'params': model.freq_conv.parameters(), 'lr': 5e-4},  # frequency features
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-3}  # classifier
    ], weight_decay=5e-4, betas=(0.9, 0.999))

    # Cosine annealing for extended training
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1.5, eta_min=1e-7)

    # Training loop with extended epoch support
    model.train()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_f1 = 0.0
    patience = 15  # Longer patience for extended training
    patience_counter = 0
    max_epochs = 75  # Support extended training

    for epoch in range(max_epochs):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, label_data in dataloader:
            inputs = inputs.to(device)
            
            # Handle mixup labels
            if isinstance(label_data, tuple) and len(label_data) == 3:
                labels_a, labels_b, lam = label_data
                labels_a, labels_b = labels_a.to(device), labels_b.to(device)
                lam = lam.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                
                # Mixup loss
                loss_a = criterion(outputs, labels_a)
                loss_b = criterion(outputs, labels_b)
                loss = (lam * loss_a + (1 - lam) * loss_b).mean()
                
                # For accuracy calculation, use primary labels
                _, predicted = torch.max(outputs.data, 1)
                total += labels_a.size(0)
                correct += (predicted == labels_a).sum().item()
            else:
                # Standard training without mixup
                labels = label_data[0].to(device)  # Extract actual labels
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0

            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_dataloader)

            # Calculate F1 score
            from sklearn.metrics import f1_score as calc_f1
            val_preds = []
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_preds.extend(preds.tolist())
            
            print(f'Epoch {epoch+1}/{max_epochs}, Train: {epoch_acc:.1f}%, Val: {val_acc:.1f}%, Loss: {val_loss:.4f}')

            # Update scheduler
            scheduler.step()

            # Save best model based on F1 score and accuracy balance
            if val_acc > best_val_acc or (abs(val_acc - best_val_acc) < 0.5 and val_loss < best_val_loss):
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), out_model + '_best')
                print(f'✓ Best model saved (Acc: {val_acc:.2f}%)')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)')
                    model.load_state_dict(torch.load(out_model + '_best'))
                    break
        else:
            print(f'Epoch {epoch+1}/{max_epochs}, Train: {epoch_acc:.1f}%')

        model.train()

    print(f'Feature fusion training complete! (Trained for {epoch+1} epochs)')

    # Final evaluation
    if val_root.exists():
        print('Evaluating on Validation set...')
        model.eval()

        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]

        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

        all_preds = []
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print('Val accuracy:', accuracy_score(all_labels, all_preds))
        try:
            print('Val ROC AUC:', roc_auc_score(all_labels, all_probs))
        except Exception:
            pass
        print(classification_report(all_labels, all_preds))

    # Save model
    torch.save(model.state_dict(), out_model)
    model_info = {
        'model_type': 'fusion',
        'state_dict_path': out_model,
        'class': 'DeepfakeFeatureFusion'
    }
    joblib.dump(model_info, out_model + '_info.pkl')
    print(f'Saved feature fusion model to {out_model}')
    print(f'Model supports real/AI image detection with extended training capabilities')


def main_fusion_dual(dataset_root, out_model, max_per_class=500, max_val_images=None):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'

    print('Using Dual-Stream Residual + ResNet Fusion')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')

    # Collect image paths
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'
    real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
    fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))

    if max_per_class:
        real_files = real_files[:max_per_class]
        fake_files = fake_files[:max_per_class]

    all_files = real_files + fake_files
    all_labels = [0] * len(real_files) + [1] * len(fake_files)

    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        RandomJPEGCompression(quality_range=(45, 95), p=0.3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(all_files, all_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    val_dataloader = None
    if val_root.exists():
        real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
        fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))
        if max_val_images:
            real_val_files = real_val_files[:max_val_images]
            fake_val_files = fake_val_files[:max_val_images]
        val_files = real_val_files + fake_val_files
        val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = DeepfakeDualStream(resnet_pretrained=True).to(device)
    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    best_val_acc = 0.0
    patience = 8
    patience_counter = 0
    best_val_loss = float('inf')
    min_delta = 1e-3
    min_delta_acc = 0.1

    max_epochs = 30
    epoch_times = []
    log_every = 50
    for epoch in range(max_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader, start=1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % log_every == 0:
                avg_loss = running_loss / max(1, batch_idx)
                acc = 100 * correct / max(1, total)
                print(
                    f'Epoch {epoch+1}/{max_epochs} | Batch {batch_idx}/{len(dataloader)} | '
                    f'Loss: {avg_loss:.4f} | Acc: {acc:.1f}%'
                , flush=True)

        epoch_loss = running_loss / max(1, len(dataloader))
        epoch_times.append(time.time() - epoch_start)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining = max_epochs - (epoch + 1)
        eta_sec = int(remaining * avg_epoch_time)
        eta_min = eta_sec // 60
        eta_rem_sec = eta_sec % 60

        # Validation with TTA
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_total += labels.size(0)
                    val_correct += (preds == labels.cpu().numpy()).sum().item()

            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_dataloader)
            print(
                f'Epoch {epoch+1}/{max_epochs}, Train Acc: {100*correct/total:.1f}%, '
                f'Loss: {epoch_loss:.4f}, Val Acc: {val_acc:.1f}%, ETA: {eta_min}m {eta_rem_sec}s'
            )

            improved = (val_loss < best_val_loss - min_delta) or (val_acc > best_val_acc + min_delta_acc)
            if improved:
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                patience_counter = 0
                torch.save(model.state_dict(), out_model + '_best_dual')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping')
                    model.load_state_dict(torch.load(out_model + '_best_dual'))
                    break
            scheduler.step(val_loss)
        else:
            print(
                f'Epoch {epoch+1}/{max_epochs}, Train Acc: {100*correct/total:.1f}%, '
                f'Loss: {epoch_loss:.4f}, ETA: {eta_min}m {eta_rem_sec}s'
            )
            scheduler.step(epoch_loss)

    torch.save(model.state_dict(), out_model)
    joblib.dump({'model_type': 'fusion_dual', 'state_dict_path': out_model, 'class': 'DeepfakeDualStream'}, out_model + '_info.pkl')
    print(f'Saved dual-stream fusion model to {out_model}')


def sample_patches(img_arr, patch_size=128, n_patches=4):
    H, W, _ = img_arr.shape
    patches = []
    for _ in range(n_patches):
        if H <= patch_size or W <= patch_size:
            # center pad/crop
            y0 = max(0, (H - patch_size) // 2)
            x0 = max(0, (W - patch_size) // 2)
        else:
            y0 = random.randint(0, H - patch_size)
            x0 = random.randint(0, W - patch_size)
        patch = img_arr[y0:y0 + patch_size, x0:x0 + patch_size]
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            ph = np.zeros((patch_size, patch_size, 3), dtype=patch.dtype)
            ph[:patch.shape[0], :patch.shape[1]] = patch
            patch = ph
        patches.append(patch)
    return patches


def image_features(path, patch_size=128, n_patches=4, augment=False):
    img = load_image(path)
    patches = sample_patches(img, patch_size=patch_size, n_patches=n_patches)
    feats = []
    # collect per-patch variances for Patch Variance Aggregation
    patch_vars = []
    for p in patches:
        for p2 in (p, np.fliplr(p)) if augment else (p,):
            g = rgb_to_gray(p2)
            res = extract_residual(g)
            # patch variance (intensity variance) for aggregation
            pv = float(np.var(g))
            patch_vars.append(pv)
            res_std = float(np.std(res))
            _, hf = fft_stats(g)
            ent = lbp_entropy(g)
            feats.append([res_std, hf, ent])
    feats = np.array(feats)
    # pool per-image: mean and std
    mean = feats.mean(axis=0)
    std = feats.std(axis=0)
    # Patch variance aggregation: include mean and std of per-patch variances
    if len(patch_vars) > 0:
        pv_mean = float(np.mean(patch_vars))
        pv_std = float(np.std(patch_vars))
    else:
        pv_mean, pv_std = 0.0, 0.0
    return np.concatenate([mean, std, [pv_mean, pv_std]])


def collect_features(folder, label, max_images=None, **kwargs):
    files = list(Path(folder).rglob('*.jpg')) + list(Path(folder).rglob('*.png'))
    files = sorted([str(x) for x in files])
    if max_images:
        files = files[:max_images]
    X = []
    y = []
    for f in files:
        try:
            feat = image_features(f, **kwargs)
            X.append(feat)
            y.append(label)
        except Exception as e:
            print(f'Failed to extract {f}: {e}')
    return X, y


def combine_multiple_datasets(dataset_roots, test_split=0.15, val_split=0.15):
    """Combine multiple datasets and create train/validation/test splits."""
    all_files = []
    all_labels = []
    
    for dataset_root in dataset_roots:
        dataset_root = Path(dataset_root)
        print(f'Loading from {dataset_root}...')
        
        # Handle different dataset structures
        if (dataset_root / 'Train').exists():
            # Standard structure with Train/Validation/Test folders
            real_folder = dataset_root / 'Train' / 'Real'
            fake_folder = dataset_root / 'Train' / 'Fake'
        else:
            # Flat structure with Real and Fake folders
            real_folder = dataset_root / 'Real'
            fake_folder = dataset_root / 'Fake'
        
        if real_folder.exists():
            real_files = list(real_folder.rglob('*.jpg')) + list(real_folder.rglob('*.png'))
            all_files.extend(real_files)
            all_labels.extend([0] * len(real_files))
            print(f'  Real: {len(real_files)} images')
        
        if fake_folder.exists():
            fake_files = list(fake_folder.rglob('*.jpg')) + list(fake_folder.rglob('*.png'))
            all_files.extend(fake_files)
            all_labels.extend([1] * len(fake_files))
            print(f'  Fake: {len(fake_files)} images')
    
    # Convert to numpy arrays for shuffling and splitting
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    # Shuffle data
    indices = np.random.permutation(len(all_files))
    all_files = all_files[indices]
    all_labels = all_labels[indices]
    
    # Create stratified train/val/test split
    from sklearn.model_selection import train_test_split
    
    # First separate test set
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_split, stratify=all_labels, random_state=42
    )
    
    # Then separate validation from training
    val_size = val_split / (1 - test_split)
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=val_size, stratify=train_labels, random_state=42
    )
    
    print(f'\nCombined dataset split:')
    print(f'  Train: {len(train_files)} images ({np.sum(train_labels==0)} real, {np.sum(train_labels==1)} fake)')
    print(f'  Val: {len(val_files)} images ({np.sum(val_labels==0)} real, {np.sum(val_labels==1)} fake)')
    print(f'  Test: {len(test_files)} images ({np.sum(test_labels==0)} real, {np.sum(test_labels==1)} fake)')
    
    return train_files, val_files, test_files, train_labels, val_labels, test_labels


def main_cnn_dual_dataset(dataset_roots, out_model, max_per_class=500, max_val_images=None):
    """Train CNN on combined dataset from multiple sources."""
    print('Using Enhanced CNN for deepfake detection (Multi-dataset training)')
    print('Features: Extended epoch support (up to 100), improved regularization, dual dataset training')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on device: {device}')
    
    # Combine datasets
    train_files, val_files, test_files, train_labels, val_labels, test_labels = combine_multiple_datasets(dataset_roots)
    
    # Apply max_per_class limit
    if max_per_class:
        real_indices = np.where(train_labels == 0)[0]
        fake_indices = np.where(train_labels == 1)[0]
        
        if len(real_indices) > max_per_class:
            real_indices = np.random.choice(real_indices, max_per_class, replace=False)
        if len(fake_indices) > max_per_class:
            fake_indices = np.random.choice(fake_indices, max_per_class, replace=False)
        
        combined_indices = np.concatenate([real_indices, fake_indices])
        train_files = train_files[combined_indices]
        train_labels = train_labels[combined_indices]
    
    all_files = train_files.tolist()
    all_labels = train_labels.tolist()
    val_files = val_files.tolist()
    val_labels = val_labels.tolist()
    
    print(f'Training on {len(all_files)} images...')
    
    # Create validation dataloader
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if val_files:
        val_dataset = ImageDataset(val_files, val_labels, transform=val_transform)
        val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    else:
        val_dataloader = None
    
    # Training transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        RandomJPEGCompression(quality_range=(50,95), p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = ImageDataset(all_files, all_labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    # Initialize model
    model = DeepfakeCNN().to(device)
    class_weights = compute_class_weights(all_labels, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    
    # Training loop
    model.train()
    best_val_acc = 0.0
    best_val_loss = float('inf')
    min_delta = 1e-3
    min_delta_acc = 0.1
    patience = 10
    patience_counter = 0
    max_epochs = 100
    
    for epoch in range(max_epochs):
        # Training phase
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total
        
        # Validation phase
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    probs = tta_predict_batch(model, inputs.cpu(), device, tta_times=4)
                    preds = (probs >= 0.5).astype(int)
                    val_total += labels.size(0)
                    val_correct += (preds == labels.cpu().numpy()).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_dataloader)
            
            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')
            
            # Early stopping
            improved = (val_loss < best_val_loss - min_delta) or (val_acc > best_val_acc + min_delta_acc)
            if improved:
                best_val_acc = max(best_val_acc, val_acc)
                best_val_loss = min(best_val_loss, val_loss)
                patience_counter = 0
                torch.save(model.state_dict(), out_model + '_best')
                print(f'✓ Best model saved')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    model.load_state_dict(torch.load(out_model + '_best'))
                    break
            scheduler.step(val_loss)
        else:
            print(f'Epoch {epoch+1}/{max_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            scheduler.step(epoch_loss)
        
        model.train()
    
    # Final evaluation
    if val_dataloader:
        print('Evaluating on combined validation set...')
        model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        print('Val accuracy:', accuracy_score(all_labels, all_preds))
        try:
            print('Val ROC AUC:', roc_auc_score(all_labels, all_probs))
        except Exception:
            pass
        print(classification_report(all_labels, all_preds))
    
    # Save model
    torch.save(model.state_dict(), out_model)
    model_info = {
        'model_type': 'cnn',
        'state_dict_path': out_model,
        'class': 'DeepfakeCNN',
        'trained_on_dual_dataset': True
    }
    joblib.dump(model_info, out_model + '_info.pkl')
    print(f'Saved CNN model trained on combined datasets to {out_model}')


def main(dataset_root, out_model, max_per_class=500, patch_size=128, patches_per_image=8, model_type='rf', n_estimators=200, scale=True, augment=True, use_gpu=False, max_val_images=None, k_folds=5):
    dataset_root = Path(dataset_root)
    train_root = dataset_root / 'Train'
    val_root = dataset_root / 'Validation'
    if not train_root.exists():
        raise FileNotFoundError(f'Train folder not found under {dataset_root}')
    # Handle CNN separately
    if model_type == 'cnn':
        main_cnn(dataset_root, out_model, max_per_class, max_val_images)
        return
    elif model_type == 'cnn_kfold':
        main_cnn_kfold(dataset_root, out_model, max_per_class, max_val_images, k_folds)
        return
    elif model_type == 'resnet':
        main_resnet(dataset_root, out_model, max_per_class, max_val_images)
        return
    elif model_type == 'fusion':
        main_fusion_improved(dataset_root, out_model, max_per_class, max_val_images)
        return
    elif model_type == 'fusion_dual':
        main_fusion_dual(dataset_root, out_model, max_per_class, max_val_images)
        return

    # Traditional ML models - collect features first
    real_folder = train_root / 'Real'
    fake_folder = train_root / 'Fake'

    print('Collecting training features...')
    Xr, yr = collect_features(real_folder, 0, max_images=max_per_class, patch_size=patch_size, n_patches=patches_per_image, augment=augment)
    Xf, yf = collect_features(fake_folder, 1, max_images=max_per_class, patch_size=patch_size, n_patches=patches_per_image, augment=augment)

    X = np.array(Xr + Xf)
    y = np.array(yr + yf)

    print(f'Training {model_type} on', X.shape[0], 'samples...')
    # Stacked meta-classifier option
    if model_type == 'stacked':
        print('Training stacked meta-classifier...')
        stacked = train_stacked_classifier(X, y, n_splits=5)
        joblib.dump(stacked, out_model)
        print('Saved stacked meta-classifier to', out_model)
        return
    # Optionally perform a randomized hyperparameter search using a fast, early-stopping boosting learner
    def tune_histgb(X_tr, y_tr):
        # use HistGradientBoosting with early stopping for faster, robust tuning
        base = HistGradientBoostingClassifier(random_state=42, early_stopping=True)
        param_dist = {
            'learning_rate': expon(scale=0.1),
            'max_iter': randint(50, 300),
            'max_leaf_nodes': randint(10, 100),
            'min_samples_leaf': randint(1, 50)
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        rs = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=20, scoring='roc_auc', cv=cv, n_jobs=-1, random_state=42, verbose=1)
        rs.fit(X_tr, y_tr)
        print('Tuning best params:', rs.best_params_)
        return rs.best_estimator_

    if model_type == 'lr':
        base = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
        clf = make_pipeline(StandardScaler(), base) if scale else base
        clf.fit(X, y)
    elif model_type == 'rf':
        base = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', n_jobs=-1, random_state=42)
        clf = make_pipeline(StandardScaler(), base) if scale else base
        clf.fit(X, y)
    elif model_type == 'gb':
        # Use XGBoost with GPU (gpu_hist for XGBoost 2.0.3)
        from xgboost import XGBClassifier
        print('Using XGBoost with RTX 2050 GPU (CUDA).')
        # Scale features first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        xgb_clf = XGBClassifier(
            tree_method='hist',
            device='cuda',
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=8,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        print(f'Training XGBoost on GPU with {n_estimators} estimators...')
        xgb_clf.fit(X_scaled, y)
        clf = make_pipeline(scaler, xgb_clf)
        print('GPU training complete!')
    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    # validation
    if val_root.exists():
        print('Evaluating on Validation set...')
        Xv, yv = [], []
        real_val = val_root / 'Real'
        fake_val = val_root / 'Fake'
        Xrv, yrv = collect_features(real_val, 0, max_images=max_val_images, patch_size=patch_size, n_patches=patches_per_image, augment=augment)
        Xfv, yfv = collect_features(fake_val, 1, max_images=max_val_images, patch_size=patch_size, n_patches=patches_per_image, augment=augment)
        Xv = np.array(Xrv + Xfv)
        yv = np.array(yrv + yfv)
        # Ensure data is on CPU for prediction (XGBoost handles GPU internally)
        Xv = Xv.astype(np.float32)
        probs = clf.predict_proba(Xv)[:, 1]
        preds = (probs >= 0.5).astype(int)
        print('Val accuracy:', accuracy_score(yv, preds))
        try:
            print('Val ROC AUC:', roc_auc_score(yv, probs))
        except Exception:
            pass
        print(classification_report(yv, preds))

    # save model
    joblib.dump(clf, out_model)
    print('Saved model to', out_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DeepfakeVsReal/Dataset', help='path to Dataset folder (contains Train/Validation/Test) or comma-separated list of datasets')
    parser.add_argument('--dual_dataset', dest='dual_dataset', action='store_true', help='train on both DeepfakeVsReal and AI vs Real img datasets')
    parser.add_argument('--out', type=str, default='model_fusion_best.joblib')
    parser.add_argument('--max_per_class', type=int, default=500)
    parser.add_argument('--patch_size', type=int, default=128)
    parser.add_argument('--patches_per_image', type=int, default=8)
    parser.add_argument('--model', type=str, default='fusion', choices=['lr', 'rf', 'gb', 'cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_dual'], help='model type: lr|rf|gb|cnn|cnn_kfold|resnet|fusion|fusion_dual')
    parser.add_argument('--stacked', dest='stacked', action='store_true', help='train a stacked meta-classifier (RF+XGB+HGB -> LR meta)')
    parser.add_argument('--n_estimators', type=int, default=200, help='n_estimators for ensemble models')
    parser.add_argument('--max_val_images', type=int, default=None, help='max validation images per class (None = all)')
    parser.add_argument('--k_folds', type=int, default=5, help='number of folds for k-fold cross-validation (used with cnn_kfold)')
    parser.add_argument('--no_scale', dest='scale', action='store_false', help='disable feature scaling')
    parser.add_argument('--augment', dest='augment', action='store_true', default=True, help='use horizontal flip augmentation for patches')
    parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', help='use GPU via XGBoost (gpu_hist) if available')
    args = parser.parse_args()
    model_type = args.model
    if getattr(args, 'stacked', False):
        model_type = 'stacked'
    
    # Handle dual dataset training
    if args.dual_dataset:
        dataset_list = ['DeepfakeVsReal/Dataset', 'AI vs Real img']
        print(f'Training on dual datasets: {dataset_list}')
        if model_type == 'cnn':
            main_cnn_dual_dataset(dataset_list, args.out, max_per_class=args.max_per_class)
        else:
            print(f'Multi-dataset training currently supported for CNN models. Use --model cnn')
    else:
        # Handle comma-separated datasets
        if ',' in args.dataset:
            dataset_list = [d.strip() for d in args.dataset.split(',')]
            print(f'Training on multiple datasets: {dataset_list}')
            if model_type == 'cnn':
                main_cnn_dual_dataset(dataset_list, args.out, max_per_class=args.max_per_class)
            else:
                print(f'Multi-dataset training currently supported for CNN models. Use --model cnn')
        else:
            main(args.dataset, args.out, max_per_class=args.max_per_class, patch_size=args.patch_size, patches_per_image=args.patches_per_image, model_type=model_type, n_estimators=args.n_estimators, scale=args.scale, augment=args.augment, use_gpu=args.use_gpu, max_val_images=args.max_val_images, k_folds=args.k_folds)
