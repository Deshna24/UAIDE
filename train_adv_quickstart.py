#!/usr/bin/env python3
"""
QUICK START GUIDE FOR train_adv.py
====================================

Advanced Deepfake Detection Training with ResNet+FFT + ConvexNet Ensemble

USAGE EXAMPLES:
───────────────

1. Train ResNet+FFT model:
   python train_adv.py --model resnet_fft --epochs 100 --batch_size 32 --use_gpu

2. Train ConvexNet (lightweight):
   python train_adv.py --model convexnet --epochs 80 --batch_size 64 --use_gpu

3. Train ENSEMBLE (ResNet+FFT + ConvexNet):
   python train_adv.py --model ensemble --epochs 120 --batch_size 32 --use_gpu

4. Custom configuration:
   python train_adv.py --model ensemble \\
     --epochs 150 \\
     --batch_size 48 \\
     --lr 0.0005 \\
     --weight_decay 1e-3 \\
     --max_per_class 2000 \\
     --use_gpu \\
     --data_dir "DeepfakeVsReal/Dataset" \\
     --output_dir "models_adv"

AVAILABLE ARGUMENTS:
────────────────────
  --model              : ensemble | resnet_fft | convexnet (default: ensemble)
  --epochs             : Number of training epochs (default: 100)
  --batch_size         : Training batch size (default: 32)
  --lr                 : Learning rate (default: 1e-3)
  --weight_decay       : L2 regularization (default: 1e-4)
  --max_per_class      : Max images per class (default: 1000)
  --use_gpu            : Use GPU for training (optional flag)
  --data_dir           : Path to dataset (default: DeepfakeVsReal/Dataset)
  --output_dir         : Directory for saved models (default: models_adv)

KEY FEATURES:
─────────────

✓ DUAL-ARCHITECTURE ENSEMBLE
  • ResNet50+FFT: Combines spatial and frequency domain features
  • ConvexNet: Lightweight, parameter-efficient architecture
  • Ensemble predictions: Average both models for robustness

✓ ADVANCED LOSS FUNCTIONS
  • Focal Loss: Focuses on hard examples, handles class imbalance
  • Label Smoothing: Prevents overconfidence

✓ DATA AUGMENTATION
  • CutMix: Blends patches between images
  • Mixup: Linear interpolation between samples
  • ColorJitter, Rotation, Gaussian Blur, Affine transforms
  • Random crops and flips

✓ OPTIMIZATIONS
  • Mixed Precision Training (AMP) for faster training
  • Exponential Moving Average (EMA) for better generalization
  • Gradient clipping to prevent exploding gradients
  • Cosine annealing with warmup LR scheduling
  • Weighted random sampling for class balance

✓ MODEL ARCHITECTURES

  ResNet+FFT Fusion:
  ├─ ResNet50 backbone (spatial features)
  ├─ FFT extractor (frequency features)
  └─ Fusion layers + Classification head

  ConvexNet (Lightweight):
  ├─ Depthwise separable convolutions
  ├─ Squeeze-and-Excitation blocks
  └─ ~500K parameters (vs 25M for ResNet)

OUTPUT FILES:
──────────────
  models_adv/
  ├─ best_model1_adv.pt (or best_model_adv.pt)
  ├─ best_model2_adv.pt (if ensemble)
  ├─ ema1_shadow.pt / ema2_shadow.pt
  ├─ ema_shadow.pt
  └─ config_adv.json (training configuration & metrics)

TRAINING TIPS:
──────────────
• GPU strongly recommended (2-3 hours on RTX 3090, 8-12 hours on CPU)
• Start with --max_per_class 500 for quick testing
• Ensemble model provides 2-4% accuracy improvement
• ConvexNet trains faster, better for mobile deployment
• Monitor validation AUC in addition to accuracy
• Save checkpoints occur automatically when validation acc improves

INFERENCE:
──────────
After training, load models with:

  import torch
  from train_adv import ResNetFFTFusion, ConvexNet

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Single model
  model = ResNetFFTFusion(num_classes=2)
  model.load_state_dict(torch.load('models_adv/best_model_adv.pt'))
  model = model.to(device).eval()

  # Or ensemble both models for higher accuracy
  model1 = ResNetFFTFusion(num_classes=2)
  model2 = ConvexNet(num_classes=2)
  model1.load_state_dict(torch.load('models_adv/best_model1_adv.pt'))
  model2.load_state_dict(torch.load('models_adv/best_model2_adv.pt'))
  model1, model2 = model1.to(device).eval(), model2.to(device).eval()

EXPECTED PERFORMANCE:
──────────────────────
• ResNet+FFT: ~95-98% accuracy on test set
• ConvexNet: ~92-96% accuracy on test set
• Ensemble: ~96-98% accuracy (more robust)

COMPARISON WITH train.py:
─────────────────────────
Feature              │ train.py  │ train_adv.py
────────────────────┼──────────┼──────────────
Models              │ Various  │ ResNet+FFT + ConvexNet
FFT Features        │ Optional │ Integrated
Focal Loss          │ Yes      │ Yes
EMA                 │ Yes      │ Yes
Ensemble Support    │ Limited  │ Full
Parameter Count     │ ~25M     │ ~25M (ResNet) + 0.5M (ConvexNet)
Training Speed      │ Medium   │ Fast (with ConvexNet)
Generalization      │ Good     │ Better (ensemble)

"""

if __name__ == '__main__':
    print(__doc__)
