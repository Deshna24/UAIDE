# UAIDE - AI Generated Image and Video Detection

UAIDE is a full deepfake detection toolkit with:
- Image fake-vs-real detection using deep models and heuristic fallback.
- Video fake-vs-real detection with temporal modeling and Grad-CAM.
- AI source classification for detected AI content: GAN vs Diffusion.
- Ethical and safety assessment with multi-check risk analysis.
- Gradio web app for interactive image/video analysis.

This README is a unified, conflict-free reference for the current repository state.

## Main Features

1. Image detection
- EfficientNet + FFT fusion support.
- Legacy CNN/ResNet/fusion model support.
- Heuristic patch-based fallback detector.
- Grad-CAM overlays for explainability.

2. Video detection
- ResNet + LSTM temporal model.
- Frame-level and video-level fake probability.
- Face-crop aware frame extraction.
- Grad-CAM on suspicious frames.

3. AI source detection
- Second-stage classifier for AI-positive samples.
- Predicts source family:
    - GAN
    - Diffusion
- Exposes both probabilities in app outputs.

4. Ethical assessment
- Face, NSFW, age/minor risk, celebrity risk.
- Metadata/tampering/watermark checks.
- Hate symbol and misleading text checks.
- Jurisdictional compliance signaling.

## Repository Layout

Top-level scripts and modules include:

- app.py: Main Gradio application for image and video analysis.
- detector.py: Heuristic patch/frequency/textural detector functions.
- ethical_assessment.py: Multi-check ethical risk assessment.
- video_model.py: Video model classes and Grad-CAM helpers.
- video_data.py: Video dataset loading and frame extraction utilities.
- train.py: Core training pipeline for classic models.
- train_adv.py: Advanced EfficientNet + FFT training pipeline.
- train_adv_quickstart.py: Faster advanced training runner.
- train_gan_vs_diffusion.py: GAN-vs-Diffusion source classifier training.
- train_video.py: Video model training.
- predict_video.py: Video inference script.
- evaluate_model.py: Evaluation entry point.
- evaluate_validation.py: Validation evaluation.
- evaluate_validation_quick.py: Quick validation evaluation.
- evaluate_validation_comprehensive.py: Comprehensive validation evaluation.
- compare_models.py: Model comparison utility.
- diagnose_misclassification.py: Error and threshold diagnostics.
- analyze_detector_bias.py: Bias analysis helper.
- tune_face_detection.py: Face detection tuning helper.
- check_gpu.py: GPU availability check.
- demo.py: Demo runner.
- demo_integrated_assessment.py: Integrated demo runner.
- test_face_detection.py: Face detection tests.
- test_integration.py: Integration tests.
- requirements.txt: Python dependencies.
- pyproject.toml: Project configuration.

Key folders:

- models_adv/: Advanced model artifacts.
- models_v2/: Alternate model artifacts.
- models_gan_vs_diffusion/: GAN-vs-Diffusion model artifacts.
- models_advanced/: Additional model outputs.
- models_adv_ensemble/: Ensemble model outputs.
- models_adv_retrain/: Retraining outputs.
- Deepfake vs Real/: Image dataset split folders.
- deepfakevsreal/: Additional dataset storage.
- SDFVD/: Video dataset storage.

## Quick Start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app opens a local Gradio interface (usually on http://127.0.0.1:7860).

## Model Loading Behavior in app.py

The app loads models in priority order:

1. Image fake-vs-real model
- Tries models_adv first.
- Falls back to models_v2 and then other discovered artifacts.

2. Video model
- Loads video_resnet_lstm.pt when available.

3. AI source model (GAN vs Diffusion)
- Loads models_gan_vs_diffusion/best_model_weights.pt + config.json when available.
- Used as second-stage classification when content is predicted AI-generated.

## Training Workflows

1. Core training

```bash
python train.py --dataset "DeepfakeVsReal/Dataset"
```

2. Advanced training

```bash
python train_adv.py --dataset "DeepfakeVsReal/Dataset"
```

3. Quick advanced training

```bash
python train_adv_quickstart.py --dataset "DeepfakeVsReal/Dataset"
```

4. GAN vs Diffusion source training

```bash
python train_gan_vs_diffusion.py --output_dir models_gan_vs_diffusion
```

5. Video training

```bash
python train_video.py --dataset "SDFVD/SDFVD" --out video_resnet_lstm.pt
```

## Evaluation and Diagnostics

Useful commands:

```bash
python evaluate_model.py
python evaluate_validation.py
python evaluate_validation_quick.py
python evaluate_validation_comprehensive.py
python compare_models.py
python diagnose_misclassification.py
python analyze_detector_bias.py
```

## App Outputs

Image and video tabs expose:

- Classification label (Real or AI-generated).
- AI-likelihood score.
- Heatmap or Grad-CAM visualization.
- Ethical status and detailed report.
- AI Source Analysis:
    - Predicted source: GAN or DIFFUSION.
    - GAN probability.
    - Diffusion probability.

## Dependencies

Install all Python requirements:

```bash
pip install -r requirements.txt
```

The repository currently expects core packages such as:

- torch
- torchvision
- gradio
- opencv-python
- scikit-learn
- scipy
- scikit-image
- Pillow
- numpy
- joblib
- datasets

## Troubleshooting

1. timm not found
- Install with: pip install timm

2. CUDA out of memory
- Use smaller batch sizes for training.
- Use a lighter backbone (for example efficientnet_b0/resnet18).

3. Source model not loaded
- Ensure both exist:
    - models_gan_vs_diffusion/best_model_weights.pt
    - models_gan_vs_diffusion/config.json

4. Video model not loaded
- Ensure video_resnet_lstm.pt exists in repository root.

5. Slow startup
- Startup can include model initialization; first launch may take longer.

## Notes on Large Files

Datasets and large model artifacts are local runtime assets and are now covered by .gitignore patterns to keep repository history clean.

## License

MIT (if a LICENSE file is present and configured accordingly).
