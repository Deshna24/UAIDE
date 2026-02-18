import io
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.transforms.functional as TF
import cv2

from detector import sliding_patch_scores, reconstruct_heatmap, rgb_to_gray, extract_residual, fft_stats, lbp_entropy
from ethical_assessment import EthicalAssessment, format_ethical_report, get_simple_status


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
AUTO_THRESHOLD = None

TARGET_REAL_FPR = 0.05
MAX_CALIB_IMAGES = 200

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
    if model_type in ['resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
        size = 224
    else:
        size = 128
    return transforms.Compose([
        transforms.Lambda(lambda img: pad_to_min_size(img, size)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def _evaluate_deep_model(model, model_type, dataset_root, max_val_images=None):
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
    if model is None or model_info is None:
        return None

    val_root = Path(dataset_root) / 'Validation'
    if not val_root.exists():
        return None

    mtype = model_info.get('model_type', 'unknown') if isinstance(model_info, dict) else 'unknown'
    real_probs = []

    if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
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
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

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

    mtype = info['model_type']
    model = None

    if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
        if mtype == 'resnet':
            from train import DeepfakeResNet as _ModelClass
        elif mtype == 'fusion_dual':
            from train import DeepfakeDualStream as _ModelClass
        elif mtype in ['fusion', 'fusion_improved']:
            from train import DeepfakeFeatureFusion as _ModelClass
        else:
            from train import DeepfakeCNN as _ModelClass

        model = _ModelClass()
        state_path = info.get('state_dict_path')
        if state_path is None:
            base = str(info_path).replace('_improved_info.pkl', '').replace('_info.pkl', '')
            candidates = [base + '_best_improved', base + '_best', base]
            for c in candidates:
                if Path(c).exists():
                    state_path = c
                    break
        if state_path is None:
            raise FileNotFoundError('state_dict_path not found in model info and no candidate file exists')

        model.load_state_dict(torch.load(state_path, map_location='cpu'))
        model.eval()
        if torch.cuda.is_available():
            model.to(torch.device('cuda'))
    else:
        base = str(info_path).replace('_improved_info.pkl', '').replace('_info.pkl', '')
        joblib_path = base + '.joblib'
        if Path(joblib_path).exists():
            model = joblib.load(joblib_path)
        else:
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


def select_best_model(dataset_root='DeepfakeVsReal/Dataset', max_val_images=None):
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

        if mtype in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
            stats = _evaluate_deep_model(model, mtype, dataset_root, max_val_images=max_val_images)
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


# Use fast loading for Gradio startup - no validation evaluation
MODEL_INFO_PATH = Path('model_fusion_best.joblib_info.pkl')
if MODEL_INFO_PATH.exists():
    try:
        MODEL, MODEL_INFO = _load_model_from_info(MODEL_INFO_PATH)
        MODEL_PATH = str(MODEL_INFO_PATH)
        print(f'=== Loaded Fixed Model ===')
        print(f"Model: {MODEL_INFO_PATH.name}")
        print(f"Type: {MODEL_INFO.get('model_type', 'unknown') if isinstance(MODEL_INFO, dict) else 'unknown'}")
    except Exception as e:
        print(f'Failed to load fixed model {MODEL_INFO_PATH.name}: {e}')
        MODEL = None
        MODEL_INFO = None
        MODEL_PATH = None
else:
    print(f'Fixed model info not found: {MODEL_INFO_PATH.name}')

if MODEL is None:
    print('No valid model loaded; falling back to heuristic detection.')
else:
    try:
        AUTO_THRESHOLD = _calibrate_threshold(
            MODEL,
            MODEL_INFO,
            dataset_root='DeepfakeVsReal/Dataset',
            target_real_fpr=TARGET_REAL_FPR,
            max_val_images=MAX_CALIB_IMAGES,
        )
        if AUTO_THRESHOLD is not None:
            print(f'Auto-threshold (target real FPR {TARGET_REAL_FPR:.2%}): {AUTO_THRESHOLD:.3f}')
    except Exception as e:
        print(f'Auto-threshold calibration failed: {e}')


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
    
def evaluate_model_on_validation(model, dataset_root='DeepfakeVsReal/Dataset'):
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


def predict_gradio(pil_img, ethical_threshold=0.5, show_raw_features=False):
    # pil_img is a PIL.Image
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    
    # Initialize ethical status
    ethical_status = "N/A"
    ethical_report = ""

    if MODEL is not None and MODEL_INFO is not None:
        try:
            if MODEL_INFO['model_type'] in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
                # Deep learning model prediction with Grad-CAM
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                # Prepare image transform
                if MODEL_INFO['model_type'] in ['resnet', 'fusion', 'fusion_improved']:
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
                    prob_fake = float(probs[0, 1])
                    
                    # Use auto-calibrated threshold to reduce false positives
                    threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else MODEL_INFO.get('optimal_threshold', 0.5)
                    pred_class = 1 if prob_fake >= threshold else 0

                label = 'AI-generated' if pred_class == 1 else 'Real (camera)'
                is_ai = pred_class == 1

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
                    ethical_status = get_simple_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob_fake, overlay_pil, ethical_status, ethical_report

            else:
                # Traditional ML model
                X = extract_image_features_from_array(img, patch_size=128, n_patches=8, random_state=0)
                if hasattr(MODEL, 'predict_proba'):
                    prob = float(MODEL.predict_proba(X)[:, 1][0])
                else:
                    pred = MODEL.predict(X)[0]
                    prob = float(pred)
                
                # Use auto-calibrated threshold to reduce false positives
                threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else MODEL_INFO.get('optimal_threshold', 0.5)
                is_ai = prob >= threshold
                label = 'AI-generated' if is_ai else 'Real (camera)'

                # Traditional heatmap
                patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
                heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
                overlay = make_overlay_pil(img, heat)

                # Perform ethical assessment if AI-generated detected
                if is_ai:
                    assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
                    ethical_status = get_simple_status(assessment)
                    ethical_report = format_ethical_report(assessment)
                    if not show_raw_features:
                        idx = ethical_report.find('\nRAW FEATURES:')
                        if idx != -1:
                            ethical_report = ethical_report[:idx]

                return label, prob, overlay, ethical_status, ethical_report

        except Exception as e:
            print(f"Model prediction failed: {e}")
            # Fall back to heuristic

    # Fallback heuristic
    patch_scores, coords, img_shape, ps, st = sliding_patch_scores(img, patch_size=128, stride=64)
    heat = reconstruct_heatmap(patch_scores, coords, img_shape, ps, st)
    overlay = make_overlay_pil(img, heat)

    ai_score = float(np.mean(heat))
    threshold = AUTO_THRESHOLD if AUTO_THRESHOLD is not None else 0.5
    is_ai = ai_score >= threshold
    label = 'AI-generated' if is_ai else 'Real (camera)'
    
    # Perform ethical assessment if AI-generated detected
    if is_ai:
        assessment = EthicalAssessment.assess(img, threshold=ethical_threshold)
        ethical_status = get_simple_status(assessment)
        ethical_report = format_ethical_report(assessment)
        if not show_raw_features:
            idx = ethical_report.find('\nRAW FEATURES:')
            if idx != -1:
                ethical_report = ethical_report[:idx]
    
    return label, ai_score, overlay, ethical_status, ethical_report


def apply_gradcam_overlay_from_pil(pil_img, model, model_type):
    """Apply Grad-CAM to PIL image for deep learning models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare transform
    if model_type in ['resnet', 'fusion', 'fusion_improved']:
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

    # Get Grad-CAM
    from train import GradCAM
    if model_type == 'resnet':
        target_layer = model.resnet.layer4[-1].conv3
    elif model_type in ['fusion', 'fusion_improved']:
        # For fusion model, use ResNet's last conv layer
        target_layer = model.resnet[7][-1].conv3  # layer4 of ResNet
    else:
        target_layer = model.conv4  # Last conv layer of custom CNN

    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate_cam(input_tensor, target_class=1)  # Focus on fake class

    # Create overlay
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize original image
    original = cv2.resize(np.array(pil_img), target_size)

    # Overlay
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay


title = "Advanced Deepfake Detection System with Ethical Assessment"

# Create balanced layout using Blocks
with gr.Blocks(title=title) as iface:
    gr.Markdown(f"""
    # {title}
    
    Upload an image to detect if it's AI-generated and assess its ethical status.
    
    **Current Model:** {MODEL_INFO['model_type'].upper() if MODEL_INFO else 'Heuristic-based'}
    """)
    
    with gr.Row():
        # Left Column - Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_image = gr.Image(type='pil', label='Upload Image', height=400)
            
            gr.Markdown("### Settings")
            ethical_threshold = gr.Slider(
                minimum=0.0, 
                maximum=1.0, 
                step=0.01, 
                value=0.5, 
                label='Ethical Risk Threshold',
                info='Lower = more strict classification'
            )
            show_raw_features = gr.Checkbox(
                label='Show raw feature values', 
                value=False
            )
            
            analyze_btn = gr.Button("Analyze Image", variant="primary", size="lg")
            
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
                detection_result = gr.Label(num_top_classes=2, label='Classification')
                ai_score = gr.Number(label='AI-likelihood Score', precision=4)
            
            heatmap = gr.Image(label='Detection Heatmap', height=400)
            
            gr.Markdown("### Ethical Assessment")
            ethical_status = gr.Textbox(label='Status', lines=2)
            
            with gr.Accordion("Full Report", open=False):
                ethical_report = gr.Textbox(
                    label='Detailed Assessment', 
                    lines=30
                )
    
    gr.Markdown("""
    ---
    **How it works:** The heatmap overlay shows regions the model considers suspicious for deepfake artifacts.
    Ethical classification is based on artifact detectability and human face presence.
    
    *Powered by FHIBE Dataset concepts for face authenticity verification.*
    """)
    
    # Connect button to function
    analyze_btn.click(
        fn=predict_gradio,
        inputs=[input_image, ethical_threshold, show_raw_features],
        outputs=[detection_result, ai_score, heatmap, ethical_status, ethical_report]
    )


if __name__ == '__main__':
    iface.launch()