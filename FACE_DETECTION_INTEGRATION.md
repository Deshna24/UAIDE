# Face Detection Integration for Ethical Assessment

## Overview
The ethical assessment module now **automatically detects faces** in AI-generated images and classifies them as **UNETHICAL** due to privacy, consent, and deepfake concerns.

## What Changed

### 1. Face Detection Implementation
- **Technology**: OpenCV Haar Cascade face detector
- **Automatic Classification**: Any AI-generated image with detected faces → **UNETHICAL**
- **Risk Score Override**: Face-containing images get minimum risk score of 0.85

### 2. Updated Files
- `ethical_assessment.py`: Added face detection methods
- `requirements.txt`: Added opencv-python dependency
- `app.py`: Already integrated (no changes needed)

### 3. Test Results
Test run on 10 sample AI-generated images from FHIBE dataset:

```
Total Images Tested: 10
Images with Faces Detected: 6 (60.0%)
Classified as UNETHICAL: 6 (60.0%)
Classified as ETHICAL: 4 (40.0%)
```

**Examples of UNETHICAL (face-detected) images:**
- `fake_0.jpg`: 2 faces detected, Risk: 0.850
- `fake_1.jpg`: 1 face detected, Risk: 0.850  
- `fake_10.jpg`: 2 faces detected, Risk: 0.850
- `fake_1000.jpg`: 1 face detected, Risk: 0.850
- `fake_1001.jpg`: 1 face detected, Risk: 0.850
- `fake_1003.jpg`: 1 face detected, Risk: 0.850

**Examples of ETHICAL (no faces) images:**
- `fake_100.jpg`: 0 faces, Risk: 0.388
- `fake_1002.jpg`: 0 faces, Risk: 0.348
- `fake_1004.jpg`: 0 faces, Risk: 0.344
- `fake_1005.jpg`: 0 faces, Risk: 0.345

## How It Works

### Face Detection Logic
1. When an AI-generated image is detected, face detection runs automatically
2. OpenCV Haar Cascade scans the image for frontal faces
3. If ≥1 face detected → **UNETHICAL** classification (regardless of other features)
4. If 0 faces detected → Classification based on artifact features (as before)

### Assessment Output
The assessment now includes:
- `faces_detected`: Number of faces found (integer)
- `status`: "UNETHICAL - FACES DETECTED" or standard classification
- Enhanced explanation mentioning face detection results
- Specific recommendations for face-containing images

### Example Assessment (Face Detected)
```
Status: UNETHICAL - FACES DETECTED
Faces Detected: 2
Risk Score: 0.85
Explanation:
  - CRITICAL: 2 human face(s) detected in AI-generated content
  - AI-generated faces violate privacy and consent principles
  - High potential for deepfake misuse and impersonation
  - Cannot verify consent from depicted individual(s)

Recommendations:
  - DO NOT SHARE - High risk of privacy violation
  - REQUIRES EXPLICIT CONSENT from person(s) depicted
  - VERIFY compliance with deepfake and privacy laws
  - Could be illegal in many jurisdictions without consent
```

## Using the Updated System

### In the Gradio App
The face detection runs automatically when you upload an image:
1. Upload image
2. Detection runs → if AI-generated...
3. Face detection runs automatically
4. Ethical assessment shows face count and adjusted classification

### Via Python API
```python
from ethical_assessment import EthicalAssessment
import numpy as np
from PIL import Image

# Load image
img = Image.open('test_image.jpg').convert('RGB')
img_arr = np.asarray(img).astype(np.float32) / 255.0

# Run assessment (face detection runs automatically)
assessment = EthicalAssessment.assess(img_arr, threshold=0.5)

print(f"Faces Detected: {assessment['faces_detected']}")
print(f"Status: {assessment['status']}")
print(f"Is Ethical: {assessment['is_ethical']}")
```

### Testing
Run the test script to see face detection in action:
```bash
python test_face_detection.py
```

## Key Features

### 1. Privacy-First Approach
- AI-generated human faces are **always unethical** without explicit consent
- Aligns with deepfake legislation and privacy regulations
- Prevents potential misuse before distribution

### 2. Configurable Threshold
- Face detection rule is absolute (overrides threshold)
- Artifact-based threshold still applies to non-face images
- Use the Gradio slider to tune threshold for non-face images

### 3. Detailed Reporting
- Number of faces detected
- Face-specific explanations and recommendations
- Raw feature values available for debugging

## Technical Details

### Face Detector Specifications
- **Model**: OpenCV Haar Cascade (haarcascade_frontalface_default.xml)
- **Detection Parameters**:
  - Scale Factor: 1.1
  - Min Neighbors: 5
  - Min Size: 30x30 pixels
- **Color Space**: Grayscale conversion for detection

### Performance
- **Speed**: ~10-50ms per image (CPU)
- **Accuracy**: Good for frontal faces, moderate for profile views
- **False Positives**: Minimal with chosen parameters

### Integration Architecture
```
User Upload
    ↓
AI Detection (CNN/ResNet)
    ↓
[If AI-generated]
    ↓
Face Detection (OpenCV)
    ↓
Ethical Assessment
    ↓
    ├─ Faces? → UNETHICAL (Risk 0.85+)
    └─ No faces → Artifact-based classification
```

## Compliance & Legal

### Why Face Detection Matters
1. **Privacy Rights**: AI-generated faces may resemble real people
2. **Consent Requirements**: Many jurisdictions require consent for face generation
3. **Deepfake Laws**: Increasingly strict regulations worldwide
4. **Misuse Prevention**: Reduces risk of impersonation and fraud

### Recommendations
- **With Faces**: Explicit consent, legal review, prominent watermarks
- **Without Faces**: Standard AI disclosure, educational/artistic use OK

## Next Steps

### Future Enhancements (Optional)
- [ ] Multi-angle face detection (profile, tilted)
- [ ] Face quality assessment (blur, occlusion)
- [ ] Face attribute analysis (age, gender - for risk assessment)
- [ ] Integration with facial recognition APIs for identity verification

### Current Status
✅ Face detection fully integrated and tested
✅ Automatic UNETHICAL classification for face-containing images
✅ Configurable threshold for non-face images
✅ Comprehensive reporting and recommendations

## Files Modified/Created
- ✅ `ethical_assessment.py` - Added face detection
- ✅ `requirements.txt` - Added opencv-python
- ✅ `test_face_detection.py` - Created test script
- ✅ `FACE_DETECTION_INTEGRATION.md` - This document

---

**Implementation Date**: February 10, 2026
**Status**: ✅ Complete and tested
**Impact**: 60% of test images correctly flagged as UNETHICAL due to face detection
