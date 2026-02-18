# UAIDE — AI-Generated Image & Video Detection

UAIDE is a deepfake detection toolkit that combines a ResNet-50 + FFT feature-fusion model with Grad-CAM explainability, ethical assessment, and a Gradio web interface. It supports both image and video analysis, with auto-calibrated thresholds to minimise false positives on real images.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Launch the web app
python app.py
```

The app loads the trained fusion model (`model_fusion_best.joblib_info.pkl`), auto-calibrates a detection threshold against the validation set, and opens a Gradio UI where you can upload images for analysis.

## How It Works

### Architecture

The primary model (`DeepfakeFeatureFusion`) fuses two streams:

1. **Spatial stream** — ResNet-50 backbone extracts high-level visual features.
2. **Frequency stream** — Block-wise FFT (16x16 blocks) produces per-block magnitude and phase statistics, processed by a small CNN. Phase information captures alignment errors that AI generators leave behind.

A **cross-attention** layer lets the spatial stream guide where to look for frequency anomalies. The fused representation passes through a classification head with progressive dropout.

### Preprocessing

Images are **padded and center-cropped** (224x224) instead of resized, preserving the original pixel-level detail that resize-based pipelines destroy.

### Training Augmentations

- Random crop, flip, rotation, colour jitter, affine
- JPEG compression (quality 50–95) and Gaussian blur — simulates real-world image degradation so the model works on compressed uploads

### Threshold Calibration

At startup, `app.py` runs the real validation images through the model and sets a threshold at the 95th percentile of their fake-class probabilities, capping the false-positive rate on real images at ~5%.

## Trained Model

The default model is stored across these files:

| File | Contents |
|------|----------|
| `model_fusion_best.joblib_info.pkl` | Model metadata (type, state-dict path, optimal threshold) |
| `model_fusion_best.joblib_best_improved` | PyTorch state dict |

Stored metrics from training:
- **Best F1**: 0.792
- **Best fake recall**: 0.681
- **Optimal threshold**: 0.371

Run `python evaluate_model.py` to compute full accuracy and ROC AUC on the Validation and Test splits.

## Training

Default settings reproduce the shipped model:

```powershell
python train.py --dataset "DeepfakeVsReal/Dataset" --max_per_class 1000
```

This trains the `fusion` model type by default and writes `model_fusion_best.joblib_best_improved` + `model_fusion_best.joblib_info.pkl`.

Other model types are available via `--model`:

| `--model` | Architecture |
|-----------|-------------|
| `fusion` (default) | ResNet-50 + block-wise FFT + cross-attention |
| `resnet` | ResNet-50 transfer learning |
| `cnn` | Custom 4-layer CNN |
| `cnn_kfold` | CNN with K-fold cross-validation |
| `fusion_dual` | Dual-stream residual + ResNet |
| `rf` | Random Forest (handcrafted features) |
| `gb` | XGBoost with GPU support |

## Video Detection

Train a ResNet-50 + LSTM on the SDFVD dataset:

```powershell
python train_video.py --dataset "SDFVD/SDFVD" --out video_resnet_lstm.pt --frames_per_video 16 --epochs 10
```

Run inference with Grad-CAM overlays:

```powershell
python predict_video.py --video path\to\video.mp4 --checkpoint video_resnet_lstm.pt
```

## Project Structure

```
UAIDE/
├── app.py                      # Gradio web interface
├── train.py                    # Training (all model types)
├── train_video.py              # Video model training
├── predict_video.py            # Video inference + Grad-CAM
├── detector.py                 # Heuristic patch-based detector
├── ethical_assessment.py        # Ethical risk scoring
├── evaluate_model.py           # Validation / Test evaluation
├── compare_models.py           # Side-by-side model comparison
├── diagnose_misclassification.py  # Threshold sweep & FP analysis
├── print_report.py             # Ethical classification report
├── video_model.py              # ResNet-LSTM architecture
├── video_data.py               # Video frame extraction
├── check_gpu.py                # GPU availability check
├── requirements.txt
├── DeepfakeVsReal/Dataset/     # Train / Validation / Test splits
├── AI vs Real img/             # Additional AI art dataset
├── SDFVD/                      # Video dataset
└── model_fusion_best.*         # Trained model artifacts
```

## Datasets

| Dataset | Location | Contents |
|---------|----------|----------|
| DeepfakeVsReal | `DeepfakeVsReal/Dataset/` | Train / Validation / Test splits with Real and Fake folders |
| AI vs Real img | `AI vs Real img/` | AI-generated art vs real art |
| SDFVD | `SDFVD/SDFVD/` | `videos_real/` and `videos_fake/` for video detection |

## Technical Details

- **Framework**: PyTorch with CUDA support
- **Backbone**: ResNet-50 (ImageNet pretrained)
- **Frequency features**: Block-wise FFT magnitude + phase (16x16 blocks, 6-channel input)
- **Attention**: Multi-head cross-attention (8 heads, 512-dim)
- **Loss**: Focal loss (alpha=0.8, gamma=2.5, label smoothing=0.15)
- **Optimiser**: AdamW with per-layer learning rates and cosine annealing
- **Input**: 224x224 center crop (no resize)
- **Regularisation**: Dropout (0.3–0.5), batch normalisation, weight decay, mixup (alpha=0.2)

## Troubleshooting

| Problem | Fix |
|---------|-----|
| CUDA out of memory | Reduce `--max_per_class` or use smaller batch size |
| Real images flagged as AI | The auto-threshold should handle this; if not, lower `TARGET_REAL_FPR` in `app.py` |
| Grad-CAM errors | Ensure `opencv-python` is installed |
| Slow startup | Threshold calibration runs on validation set at launch; reduce `MAX_CALIB_IMAGES` in `app.py` |

## Repository

https://github.com/Deshnaa2007/UAIDE
