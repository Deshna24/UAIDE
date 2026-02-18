"""
Test Face Detection in Ethical Assessment
Demonstrates how AI-generated images with faces are automatically classified as UNETHICAL
"""

import numpy as np
from PIL import Image
from pathlib import Path
from ethical_assessment import EthicalAssessment, format_ethical_report, get_simple_status

def test_face_detection():
    """Test face detection on sample AI-generated images"""
    
    print("\n" + "="*80)
    print("FACE DETECTION ETHICAL ASSESSMENT TEST")
    print("="*80)
    print("\nTesting face detection on AI-generated images from FHIBE dataset...")
    print("Any image with detected faces will be classified as UNETHICAL.\n")
    
    # Get sample fake images
    fake_folder = Path('DeepfakeVsReal/Dataset/Test/Fake')
    
    if not fake_folder.exists():
        print(f"Error: Fake folder not found at {fake_folder}")
        return
    
    # Get first 10 images for testing
    fake_images = list(fake_folder.glob('*.jpg'))[:10]
    
    if not fake_images:
        print("No test images found!")
        return
    
    print(f"Found {len(fake_images)} sample images to test\n")
    
    results = []
    
    for idx, img_path in enumerate(fake_images, 1):
        try:
            print(f"\n{'='*80}")
            print(f"Image {idx}/{len(fake_images)}: {img_path.name}")
            print(f"{'='*80}")
            
            # Load image
            pil_img = Image.open(img_path).convert('RGB')
            img_arr = np.asarray(pil_img).astype(np.float32) / 255.0
            
            # Run ethical assessment
            assessment = EthicalAssessment.assess(img_arr, threshold=0.5)
            
            # Print summary
            print(f"\nStatus: {assessment['status']}")
            print(f"Faces Detected: {assessment['faces_detected']}")
            print(f"Risk Score: {assessment['risk_score']:.4f}")
            print(f"Is Ethical: {assessment['is_ethical']}")
            
            # Store result
            results.append({
                'image': img_path.name,
                'faces': assessment['faces_detected'],
                'ethical': assessment['is_ethical'],
                'risk_score': assessment['risk_score']
            })
            
            # Show detailed report for images with faces
            if assessment['faces_detected'] > 0:
                print("\n" + "!"*80)
                print("FACE DETECTED - Classified as UNETHICAL")
                print("!"*80)
                print(f"Explanation: {assessment['explanation']}")
                
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            continue
    
    # Summary statistics
    print("\n\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total = len(results)
    if total == 0:
        print("\nNo images were successfully processed!")
        return
    
    with_faces = sum(1 for r in results if r['faces'] > 0)
    unethical = sum(1 for r in results if not r['ethical'])
    
    print(f"\nTotal Images Tested: {total}")
    print(f"Images with Faces Detected: {with_faces} ({with_faces/total*100:.1f}%)")
    print(f"Classified as UNETHICAL: {unethical} ({unethical/total*100:.1f}%)")
    print(f"Classified as ETHICAL: {total - unethical} ({(total-unethical)/total*100:.1f}%)")
    
    # List images with faces
    if with_faces > 0:
        print(f"\n{'='*80}")
        print("IMAGES WITH DETECTED FACES (UNETHICAL):")
        print(f"{'='*80}")
        for r in results:
            if r['faces'] > 0:
                print(f"  • {r['image']}: {r['faces']} face(s), Risk: {r['risk_score']:.3f}")
    
    print("\n" + "="*80)
    print("Face detection ethical assessment test complete!")
    print("="*80 + "\n")

if __name__ == '__main__':
    test_face_detection()
