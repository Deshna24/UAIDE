"""
Simple Demo: Face Detection + Ethical Assessment
Shows the integration working end-to-end without Gradio
"""

import numpy as np
from PIL import Image
from pathlib import Path
from ethical_assessment import EthicalAssessment

def demo_integrated_assessment():
    """
    Demonstrates the complete workflow:
    1. Load AI-generated image
    2. Run ethical assessment (with automatic face detection)
    3. Show results
    """
    
    print("\n" + "="*80)
    print("INTEGRATED ETHICAL ASSESSMENT DEMO")
    print("Face Detection + Risk Assessment")
    print("="*80 + "\n")
    
    # Test on a few sample images
    fake_folder = Path('DeepfakeVsReal/Dataset/Test/Fake')
    
    if not fake_folder.exists():
        print(f"Error: {fake_folder} not found")
        return
    
    test_images = list(fake_folder.glob('*.jpg'))[:5]
    
    print(f"Testing {len(test_images)} AI-generated images...\n")
    
    for idx, img_path in enumerate(test_images, 1):
        print(f"\n{'='*80}")
        print(f"TEST {idx}: {img_path.name}")
        print(f"{'='*80}")
        
        try:
            # Load image
            pil_img = Image.open(img_path).convert('RGB')
            img_arr = np.asarray(pil_img).astype(np.float32) / 255.0
            
            # Run ethical assessment with face detection
            assessment = EthicalAssessment.assess(img_arr, threshold=0.5)
            
            # Display results
            print(f"\nRESULTS:")
            print(f"  Faces Detected: {assessment['faces_detected']}")
            print(f"  Risk Score: {assessment['risk_score']:.4f}")
            print(f"  Threshold: {assessment['threshold']:.2f}")
            print(f"  Classification: {assessment['status']}")
            print(f"  Is Ethical: {assessment['is_ethical']}")
            print(f"  Confidence: {assessment['confidence']*100:.1f}%")
            
            # Show explanation
            print(f"\nEXPLANATION:")
            for line in assessment['explanation'].split('\n'):
                print(f"  {line}")
            
            # Show key recommendation
            print(f"\nKEY RECOMMENDATION:")
            print(f"  {assessment['recommendations'][0]}")
            
            # Highlight face detections
            if assessment['faces_detected'] > 0:
                print(f"\n{'!'*80}")
                print(f"  WARNING: This image contains {assessment['faces_detected']} face(s)")
                print(f"  Automatically classified as UNETHICAL")
                print(f"{'!'*80}")
            else:
                print(f"\n  No faces detected - artifact-based classification applied")
                
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nSUMMARY:")
    print("- Face detection runs automatically on all AI-generated images")
    print("- Images with faces are classified as UNETHICAL (Risk >= 0.85)")
    print("- Images without faces use artifact-based risk assessment")
    print("- Configurable threshold for non-face classification")
    print("\n")

if __name__ == '__main__':
    demo_integrated_assessment()
