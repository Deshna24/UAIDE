"""
Analyze detector.py's behavior on validation set to find threshold/bias issues.
"""
import argparse
import numpy as np
from pathlib import Path
from detector import process_image
from sklearn.metrics import confusion_matrix, classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--invert', action='store_true', help='invert heatmap/score before analysis')
args = parser.parse_args()

DATASET = Path('DeepfakeVsReal/Dataset')
val_root = DATASET / 'Validation'

print("=" * 80)
print("DETECTOR.PY BIAS ANALYSIS")
print("=" * 80)

# Test on sample real and fake images
real_dir = val_root / 'Real'
fake_dir = val_root / 'Fake'

# Sample 50 real and 50 fake
real_files = sorted(list(real_dir.glob('*.jpg')))[:50]
fake_files = sorted(list(fake_dir.glob('*.jpg')))[:50]

print(f"\n📊 Testing on {len(real_files)} real + {len(fake_files)} fake images...\n")

real_scores = []
fake_scores = []

# Process real images
print("👉 Real images:", end=" ", flush=True)
for i, f in enumerate(real_files):
    try:
        result = process_image(str(f), invert=args.invert)
        score = result['ai_score']
        real_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"{i+1}", end=" ", flush=True)
    except Exception as e:
        print(f"\nError on {f}: {e}")
print(f"✓ Done\n")

# Process fake images  
print("👉 Fake images:", end=" ", flush=True)
for i, f in enumerate(fake_files):
    try:
        result = process_image(str(f), invert=args.invert)
        score = result['ai_score']
        fake_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"{i+1}", end=" ", flush=True)
    except Exception as e:
        print(f"\nError on {f}: {e}")
print(f"✓ Done\n")

real_scores = np.array(real_scores)
fake_scores = np.array(fake_scores)

print("=" * 80)
print("📈 SCORE STATISTICS")
print("=" * 80)
print(f"\nReal images (should be LOW scores):")
print(f"  Mean: {real_scores.mean():.4f}")
print(f"  Std:  {real_scores.std():.4f}")
print(f"  Min:  {real_scores.min():.4f}")
print(f"  Max:  {real_scores.max():.4f}")
print(f"  Median: {np.median(real_scores):.4f}")

print(f"\nFake images (should be HIGH scores):")
print(f"  Mean: {fake_scores.mean():.4f}")
print(f"  Std:  {fake_scores.std():.4f}")
print(f"  Min:  {fake_scores.min():.4f}")
print(f"  Max:  {fake_scores.max():.4f}")
print(f"  Median: {np.median(fake_scores):.4f}")

# Check overlap
print(f"\n⚠️  OVERLAP ANALYSIS:")
all_scores = np.concatenate([real_scores, fake_scores])
true_labels = np.array([0] * len(real_scores) + [1] * len(fake_scores))

# Find best threshold (minimize false positives while keeping some recall)
best_thresh = 0.5
best_fp_rate = 1.0
best_recall = 0.0
min_recall = 0.2
for thresh in np.arange(0.0, 1.01, 0.05):
    preds = (all_scores >= thresh).astype(int)
    cm = confusion_matrix(true_labels, preds)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        continue
    
    fp_rate = fp / (tn + fp) if (tn + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if recall >= min_recall:
        if fp_rate < best_fp_rate or (fp_rate == best_fp_rate and recall > best_recall):
            best_fp_rate = fp_rate
            best_recall = recall
            best_thresh = thresh

print(f"  Overlap range: [{min(real_scores.min(), fake_scores.min()):.4f}, {max(real_scores.max(), fake_scores.max()):.4f}]")

# Test different thresholds
print(f"\n{'-'*80}")
print(f"{'Threshold':<12} {'Accuracy':<12} {'Real→AI %':<12} {'AI→Real %':<12}")
print(f"{'-'*80}")

for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds = (all_scores >= thresh).astype(int)
    cm = confusion_matrix(true_labels, preds)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, sum(preds), 0, 0
    
    acc = (tn + tp) / len(all_scores)
    real_to_ai = fp / len(real_scores) if len(real_scores) > 0 else 0  # False positive rate
    ai_to_real = fn / len(fake_scores) if len(fake_scores) > 0 else 0  # False negative rate
    
    marker = "✓ BEST" if thresh == best_thresh else ""
    print(f"{thresh:<12.2f} {acc:<12.4f} {real_to_ai:<12.2%} {ai_to_real:<12.2%}  {marker}")

print(f"\n{'-'*80}")
print(f"\n🎯 RECOMMENDATION:")
print(f"  Best threshold (min FP, recall >= {min_recall:.0%}): {best_thresh:.2f}")
print(f"  Expected false positive rate (Real→AI): ~{(all_scores[true_labels==0] >= best_thresh).mean():.2%}")
print(f"  Expected false negative rate (AI→Real): ~{(all_scores[true_labels==1] < best_thresh).mean():.2%}")

if real_scores.mean() > fake_scores.mean():
    print(f"\n❌ CRITICAL ISSUE DETECTED:")
    print(f"   Real images have HIGHER ai_score ({real_scores.mean():.4f}) than fake ({fake_scores.mean():.4f})")
    print(f"   This suggests the score should be inverted.")
    print(f"\n   SOLUTION: Run this tool with --invert, or use --invert in detector.py.")

