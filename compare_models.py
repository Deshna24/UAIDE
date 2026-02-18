#!/usr/bin/env python3
"""
Model Comparison Script for Deepfake Detection
Compares performance of different models on the validation set
"""

import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from torchvision import transforms
from PIL import Image
import pandas as pd
from train import DeepfakeCNN, DeepfakeResNet, DeepfakeFeatureFusion, ImageDataset
from torch.utils.data import DataLoader

def load_model(model_path):
    """Load a trained model based on its info file"""
    try:
        info_path = model_path + '_info.pkl'
        if not Path(info_path).exists():
            return None, None

        model_info = joblib.load(info_path)

        if model_info['model_type'] in ['cnn', 'cnn_kfold']:
            model = DeepfakeCNN()
            model.load_state_dict(torch.load(model_info['state_dict_path']))
        elif model_info['model_type'] == 'resnet':
            model = DeepfakeResNet()
            model.load_state_dict(torch.load(model_info['state_dict_path']))
        elif model_info['model_type'] == 'fusion':
            model = DeepfakeFeatureFusion()
            model.load_state_dict(torch.load(model_info['state_dict_path']))
        else:
            # Traditional ML model
            model = joblib.load(model_path)

        if isinstance(model, nn.Module):
            model.eval()
            if torch.cuda.is_available():
                model.cuda()

        return model, model_info

    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
        return None, None

def evaluate_model_on_validation(model, model_info, val_files, val_labels):
    """Evaluate a model on validation data"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_info['model_type'] in ['cnn', 'cnn_kfold', 'resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
        # Deep learning model
        if model_info['model_type'] in ['resnet', 'fusion', 'fusion_improved', 'fusion_dual']:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        val_dataset = ImageDataset(val_files, val_labels, transform=transform)
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

    else:
        # Traditional ML model - extract features
        from train import image_features
        X_val = []
        for img_path in val_files:
            try:
                features = image_features(str(img_path), patch_size=128, n_patches=8, augment=False)
                X_val.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

        X_val = np.array(X_val)
        y_val = np.array(val_labels[:len(X_val)])  # Match lengths

        # Get predictions
        if hasattr(model, 'predict_proba'):
            all_probs = model.predict_proba(X_val)[:, 1]
        else:
            all_probs = model.predict(X_val)

        all_preds = (all_probs >= 0.5).astype(int)
        all_labels = y_val

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5

    return {
        'accuracy': accuracy,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'true_labels': all_labels
    }

def main():
    print("=== Deepfake Detection Model Comparison ===\n")

    # Load validation data
    dataset_root = Path('DeepfakeVsReal/Dataset')
    val_root = dataset_root / 'Validation'

    if not val_root.exists():
        print("Validation data not found!")
        return

    real_val_files = list((val_root / 'Real').rglob('*.jpg')) + list((val_root / 'Real').rglob('*.png'))
    fake_val_files = list((val_root / 'Fake').rglob('*.jpg')) + list((val_root / 'Fake').rglob('*.png'))

    # Limit validation samples for faster comparison
    max_val_samples = 500
    real_val_files = real_val_files[:max_val_samples//2]
    fake_val_files = fake_val_files[:max_val_samples//2]

    val_files = real_val_files + fake_val_files
    val_labels = [0] * len(real_val_files) + [1] * len(fake_val_files)

    print(f"Evaluating on {len(val_files)} validation images ({len(real_val_files)} real, {len(fake_val_files)} fake)")

    # Models to compare
    model_files = [
        ('model_fusion_best.joblib', 'ResNet + FFT Fusion (Improved)'),
        ('model_dual_dataset.pt', 'CNN Dual-Dataset'),
    ]

    results = []

    for model_file, model_name in model_files:
        if not Path(model_file).exists():
            print(f"⚠️  {model_name}: Model file not found")
            continue

        print(f"\n🔍 Evaluating {model_name}...")

        model, model_info = load_model(model_file)
        if model is None:
            print(f"❌ Failed to load {model_name}")
            continue

        try:
            metrics = evaluate_model_on_validation(model, model_info, val_files, val_labels)

            print(".3f")
            print(".3f")

            results.append({
                'Model': model_name,
                'Type': model_info['model_type'] if model_info else 'unknown',
                'Accuracy': metrics['accuracy'],
                'AUC': metrics['auc']
            })

        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")

    if not results:
        print("No models could be evaluated!")
        return

    # Display comparison table
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)

    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)

    print(df.to_string(index=False, float_format='%.3f'))

    # Best model
    best_model = df.iloc[0]
    print(f"\n🏆 Best performing model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.1f}%)")

    # Save results
    df.to_csv('model_comparison_results.csv', index=False)
    print("📊 Results saved to model_comparison_results.csv")

if __name__ == '__main__':
    main()</content>
<parameter name="filePath">c:\Users\DESHNA\Desktop\UAIDE\compare_models.py