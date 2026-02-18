"""Evaluate the trained fusion model on Validation and Test sets."""
import joblib
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF

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
model = DeepfakeFeatureFusion()
model.load_state_dict(torch.load(state_path, map_location='cpu'))
model.to(device)
model.eval()
print(f'Loaded fusion_improved model from {state_path}')

transform = transforms.Compose([
    transforms.Lambda(lambda img: pad_to_min_size(img, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def evaluate_split(split_name, split_root):
    if not split_root.exists():
        print(f'No {split_name} folder found at {split_root}')
        return
    real_files = sorted([str(x) for x in (split_root / 'Real').rglob('*.jpg')] +
                        [str(x) for x in (split_root / 'Real').rglob('*.png')])
    fake_files = sorted([str(x) for x in (split_root / 'Fake').rglob('*.jpg')] +
                        [str(x) for x in (split_root / 'Fake').rglob('*.png')])
    files = real_files + fake_files
    labels = [0] * len(real_files) + [1] * len(fake_files)
    if not files:
        print(f'No {split_name} images found')
        return
    dataset = ImageDataset(files, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    all_probs, all_labels = [], []
    with torch.no_grad():
        for inputs, lbls in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(lbls.tolist())
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)
    print(f'\n{split_name} samples: {len(y_true)}')
    print(f'{split_name} accuracy: {accuracy_score(y_true, y_pred):.4f}')
    try:
        print(f'{split_name} ROC AUC: {roc_auc_score(y_true, y_prob):.4f}')
    except Exception:
        pass
    print(classification_report(y_true, y_pred))


evaluate_split('Validation', DATASET / 'Validation')
evaluate_split('Test', DATASET / 'Test')
