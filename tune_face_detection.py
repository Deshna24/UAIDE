"""
Diagnostic script to test and tune face detection parameters
"""

import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
import cv2

def test_face_detection_parameters():
    """Test different face detection parameters to find optimal settings"""
    
    print("\n" + "="*80)
    print("FACE DETECTION PARAMETER TUNING")
    print("="*80 + "\n")
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Test on sample images
    fake_folder = Path('DeepfakeVsReal/Dataset/Test/Fake')
    
    if not fake_folder.exists():
        print(f"Error: {fake_folder} not found")
        return
    
    test_images = list(fake_folder.glob('*.jpg'))[:10]
    
    print(f"Testing on {len(test_images)} images with different parameters...\n")
    
    # Different parameter combinations
    param_sets = [
        {"name": "Current (Conservative)", "scaleFactor": 1.1, "minNeighbors": 5, "minSize": (30, 30)},
        {"name": "Relaxed", "scaleFactor": 1.05, "minNeighbors": 3, "minSize": (20, 20)},
        {"name": "Very Relaxed", "scaleFactor": 1.03, "minNeighbors": 2, "minSize": (15, 15)},
        {"name": "Strict", "scaleFactor": 1.2, "minNeighbors": 6, "minSize": (40, 40)},
    ]
    
    results = {}
    
    for params in param_sets:
        param_name = params["name"]
        results[param_name] = []
        
        print(f"\n{'='*80}")
        print(f"Testing: {param_name}")
        print(f"  scaleFactor={params['scaleFactor']}, minNeighbors={params['minNeighbors']}, minSize={params['minSize']}")
        print(f"{'='*80}")
        
        for idx, img_path in enumerate(test_images, 1):
            try:
                # Load and convert image
                pil_img = Image.open(img_path).convert('RGB')
                img_arr = np.asarray(pil_img).astype(np.float32) / 255.0
                
                # Convert to grayscale
                gray = cv2.cvtColor((img_arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=params['scaleFactor'],
                    minNeighbors=params['minNeighbors'],
                    minSize=params['minSize']
                )
                
                num_faces = len(faces)
                results[param_name].append(num_faces)
                
                print(f"  {img_path.name}: {num_faces} faces")
                
                # Optionally save annotated image for first few
                if idx <= 3 and num_faces > 0:
                    # Draw rectangles around faces
                    img_with_boxes = pil_img.copy()
                    draw = ImageDraw.Draw(img_with_boxes)
                    for (x, y, w, h) in faces:
                        draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                    
                    output_path = Path(f'face_detection_test_{param_name.replace(" ", "_")}_{img_path.name}')
                    img_with_boxes.save(output_path)
                    print(f"    Saved annotated image: {output_path}")
                    
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
                continue
    
    # Summary comparison
    print("\n\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80 + "\n")
    
    for param_name, face_counts in results.items():
        total = sum(face_counts)
        avg = total / len(face_counts) if face_counts else 0
        images_with_faces = sum(1 for c in face_counts if c > 0)
        
        print(f"{param_name}:")
        print(f"  Total faces detected: {total}")
        print(f"  Average per image: {avg:.2f}")
        print(f"  Images with faces: {images_with_faces}/{len(face_counts)} ({images_with_faces/len(face_counts)*100:.1f}%)")
        print()
    
    print("="*80)
    print("Recommendation: Use 'Relaxed' or 'Very Relaxed' for better detection")
    print("="*80 + "\n")

if __name__ == '__main__':
    test_face_detection_parameters()
