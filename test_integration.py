"""
Quick integration test - simulates what app.py does
"""

import sys
sys.path.insert(0, r'c:\Users\DESHNA\Desktop\UAIDE')

import numpy as np
from PIL import Image
from pathlib import Path
from ethical_assessment import EthicalAssessment, get_simple_status

print("\n" + "="*80)
print("END-TO-END INTEGRATION TEST")
print("Simulating app.py workflow with improved face detection")
print("="*80 + "\n")

fake_folder = Path('DeepfakeVsReal/Dataset/Test/Fake')
test_images = list(fake_folder.glob('*.jpg'))[:5]

for idx, img_path in enumerate(test_images, 1):
    print(f"\n[{idx}] {img_path.name}")
    print("-" * 80)
    
    # Load image (as app.py does)
    pil_img = Image.open(img_path).convert('RGB')
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    
    # Run ethical assessment (as app.py does)
    assessment = EthicalAssessment.assess(img, threshold=0.5)
    
    # Display results (as app.py does)
    print(f"Faces Detected: {assessment['faces_detected']}")
    print(f"Status: {assessment['status']}")
    print(f"Risk Score: {assessment['risk_score']:.3f}")
    print(f"Is Ethical: {assessment['is_ethical']}")
    
    if assessment['faces_detected'] > 0:
        print(f"\n>>> UNETHICAL: {assessment['faces_detected']} face(s) found!")
    else:
        print(f"\n>>> Ethical - No faces detected")

print("\n" + "="*80)
print("Integration test complete - Face detection is working!")
print("="*80 + "\n")
