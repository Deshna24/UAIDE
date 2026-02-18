"""
Diagnose misclassification issue by analyzing predicted probabilities,
finding optimal threshold, and showing false positive examples.
"""
import joblib
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, roc_curve
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt

from train import DeepfakeFeatureFusion, ImageDataset


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


MODEL_INFO_PATH = Path('model_fusion_best.joblib_info.pkl')
DATASET = Path('DeepfakeVsReal/Dataset')

if not MODEL_INFO_PATH.exists():
    raise FileNotFoundError(f'{MODEL_INFO_PATH} not found')

model_info = joblib.load(str(MODEL_INFO_PATH))
state_path = model_info.get('state_dict_path')
if state_path is None or not Path(state_path).exists():
    raise FileNotFoundError(f'State dict not found: {state_path}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf = DeepfakeFeatureFusion()
clf.load_state_dict(torch.load(state_path, map_location='cpu'))
clf.to(device)
clf.eval()

transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print('=' * 80)
print('DEEPFAKE DETECTION DIAGNOSTIC')
print('=' * 80)
print(f'\n✓ Loaded fusion_improved model from {state_path}')

# Validation set
val_root = DATASET / 'Validation'
print(f'\nEvaluating on Validation set...')
real_val = val_root / 'Real'
fake_val = val_root / 'Fake'

real_files = sorted([str(x) for x in real_val.rglob('*.jpg')] + [str(x) for x in real_val.rglob('*.png')])
fake_files = sorted([str(x) for x in fake_val.rglob('*.jpg')] + [str(x) for x in fake_val.rglob('*.png')])
files = real_files + fake_files
labels = [0] * len(real_files) + [1] * len(fake_files)

if len(files) > 0:
    dataset = ImageDataset(files, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = clf(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.tolist())
    yv = np.array(all_labels)
    probs_v = np.array(all_probs)
    
    print(f'  Total samples: {len(yv)}')
    print(f'  Real images: {sum(yv == 0)}')
    print(f'  Fake images: {sum(yv == 1)}')
    
    # Current threshold (0.5)
    preds_05 = (probs_v >= 0.5).astype(int)
    acc_05 = accuracy_score(yv, preds_05)
    cm_05 = confusion_matrix(yv, preds_05)
    tn, fp, fn, tp = cm_05.ravel()
    
    print(f'\n--- Current Threshold: 0.5 ---')
    print(f'  Accuracy: {acc_05:.4f}')
    print(f'  True Negatives (Real correctly as Real): {tn}')
    print(f'  False Positives (Real wrongly as Fake): {fp} ← PROBLEM')
    print(f'  False Negatives (Fake wrongly as Real): {fn}')
    print(f'  True Positives (Fake correctly as Fake): {tp}')
    
    # Try to compute ROC AUC
    try:
        roc_auc = roc_auc_score(yv, probs_v)
        print(f'  ROC AUC: {roc_auc:.4f}')
    except Exception as e:
        print(f'  ROC AUC: Error - {e}')
    
    # Find best threshold to minimize false positives for real images
    # (maximize specificity while maintaining reasonable sensitivity)
    print(f'\n--- Threshold Sweep (Finding Optimal) ---')
    thresholds = np.arange(0.1, 1.0, 0.05)
    best_threshold = 0.5
    best_fp_rate = 1.0
    best_metrics = {}
    
    for thresh in thresholds:
        preds = (probs_v >= thresh).astype(int)
        cm = confusion_matrix(yv, preds)
        tn, fp, fn, tp = cm.ravel()
        
        # False positive rate for real images (we want this LOW)
        fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
        acc = accuracy_score(yv, preds)
        
        status = "✓" if fp_rate < best_fp_rate else " "
        print(f'  {status} Threshold {thresh:.2f}: Acc={acc:.4f}, FP_Rate={fp_rate:.4f} (FP={fp}, TP={tp})')
        
        if fp_rate < best_fp_rate:
            best_fp_rate = fp_rate
            best_threshold = thresh
            best_metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'fp_rate': fp_rate, 'acc': acc}
    
    print(f'\n--- RECOMMENDATION ---')
    print(f'Best threshold: {best_threshold:.2f}')
    print(f'  Expected metrics:')
    print(f'    False Positive Rate: {best_metrics["fp_rate"]:.4f}')
    print(f'    Accuracy: {best_metrics["acc"]:.4f}')
    print(f'    Real→Fake (FP): {best_metrics["fp"]}')
    print(f'    Fake→Real (FN): {best_metrics["fn"]}')
    
    # Show detailed classification report at best threshold
    preds_best = (probs_v >= best_threshold).astype(int)
    print(f'\n--- Classification Report (Threshold {best_threshold:.2f}) ---')
    print(classification_report(yv, preds_best, target_names=['Real', 'Fake']))

# Test set
test_root = DATASET / 'Test'
if test_root.exists():
    print(f'\n{"=" * 80}')
    print(f'Evaluating on Test set...')
    real_test = test_root / 'Real'
    fake_test = test_root / 'Fake'
    real_t = sorted([str(x) for x in real_test.rglob('*.jpg')] + [str(x) for x in real_test.rglob('*.png')])
    fake_t = sorted([str(x) for x in fake_test.rglob('*.jpg')] + [str(x) for x in fake_test.rglob('*.png')])
    files_t = real_t + fake_t
    labels_t = [0] * len(real_t) + [1] * len(fake_t)
    
    if len(files_t) > 0:
        ds_t = ImageDataset(files_t, labels_t, transform=transform)
        dl_t = DataLoader(ds_t, batch_size=8, shuffle=False, num_workers=0)
        all_probs_t, all_labels_t = [], []
        with torch.no_grad():
            for inputs, lbls in dl_t:
                inputs = inputs.to(device)
                outputs = clf(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs_t.extend(probs.tolist())
                all_labels_t.extend(lbls.tolist())
        yt = np.array(all_labels_t)
        probs_t = np.array(all_probs_t)
        
        print(f'  Total samples: {len(yt)}')
        print(f'  Real images: {sum(yt == 0)}')
        print(f'  Fake images: {sum(yt == 1)}')
        
        # Test with best threshold
        preds_best = (probs_t >= best_threshold).astype(int)
        acc_best = accuracy_score(yt, preds_best)
        cm_best = confusion_matrix(yt, preds_best)
        tn, fp, fn, tp = cm_best.ravel()
        
        print(f'\nTest Results (Threshold {best_threshold:.2f}):')
        print(f'  Accuracy: {acc_best:.4f}')
        print(f'  Real→Fake (False Positive): {fp}')
        print(f'  Fake→Real (False Negative): {fn}')
        print(f'\n{classification_report(yt, preds_best, target_names=["Real", "Fake"])}')

print(f'\n{"=" * 80}')
print('NEXT STEPS:')
print('1. Update threshold in your prediction scripts from 0.5 to recommended value')
print('2. If false positives still too high, consider:')
print('   - Retraining with adjusted class weights')
print('   - Adding more diverse real image samples')
print('   - Checking for preprocessing mismatches')
print('=' * 80)
