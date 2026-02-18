import pandas as pd
import numpy as np

df = pd.read_csv('ethical_classification_results.csv')

print("=" * 80)
print("FHIBE ETHICAL CLASSIFICATION REPORT - AI-GENERATED IMAGES")
print("=" * 80)

print(f"\n📊 DATASET SUMMARY:")
print(f"   Total Images Analyzed: {len(df)}")

ethical_count = len(df[df['ethical_label'] == 0])
unethical_count = len(df[df['ethical_label'] == 1])

print(f"   ✓ ETHICAL (Detectable/Safe): {ethical_count} ({ethical_count/len(df)*100:.1f}%)")
print(f"   ✗ UNETHICAL (Convincing/Risky): {unethical_count} ({unethical_count/len(df)*100:.1f}%)")

print(f"\n📈 RISK SCORE STATISTICS:")
print(f"   Mean: {df['risk_score'].mean():.4f}")
print(f"   Median: {df['risk_score'].median():.4f}")
print(f"   Std Dev: {df['risk_score'].std():.4f}")
print(f"   Range: [{df['risk_score'].min():.4f}, {df['risk_score'].max():.4f}]")

print(f"\n🔍 FEATURE ANALYSIS:")
features = ['high_quality_artifacts', 'low_quality_artifacts', 
           'facial_consistency', 'lighting_anomalies', 'frequency_analysis']
f_labels = ['HQ Artifacts', 'LQ Score', 'Facial Consistency', 'Lighting', 'Frequency']

for label, col in zip(f_labels, features):
    print(f"   {label}: Mean={df[col].mean():.4f}, Std={df[col].std():.4f}")

print(f"\n📋 DETAILED BREAKDOWN:")

if ethical_count > 0:
    eth_df = df[df['ethical_label'] == 0]
    print(f"\n   ETHICAL IMAGES (Safe/Detectable):")
    print(f"      Count: {ethical_count}")
    print(f"      Mean Risk: {eth_df['risk_score'].mean():.4f}")
    print(f"      Mean Artifact Std: {eth_df['artifact_std'].mean():.4f}")
    print(f"      Mean Facial Consistency: {eth_df['facial_consistency'].mean():.4f}")

if unethical_count > 0:
    uneth_df = df[df['ethical_label'] == 1]
    print(f"\n   UNETHICAL IMAGES (Risky/Convincing):")
    print(f"      Count: {unethical_count}")
    print(f"      Mean Risk: {uneth_df['risk_score'].mean():.4f}")
    print(f"      Mean Artifact Std: {uneth_df['artifact_std'].mean():.4f}")
    print(f"      Mean Facial Consistency: {uneth_df['facial_consistency'].mean():.4f}")

print(f"\n⚠️  RISK DISTRIBUTION:")
ranges = {
    'Very Low (0.0-0.2)': len(df[df['risk_score'] <= 0.2]),
    'Low (0.2-0.4)': len(df[(df['risk_score'] > 0.2) & (df['risk_score'] <= 0.4)]),
    'Medium (0.4-0.6)': len(df[(df['risk_score'] > 0.4) & (df['risk_score'] <= 0.6)]),
    'High (0.6-0.8)': len(df[(df['risk_score'] > 0.6) & (df['risk_score'] <= 0.8)]),
    'Critical (0.8-1.0)': len(df[df['risk_score'] > 0.8])
}

for level, count in ranges.items():
    pct = count / len(df) * 100
    bar = '█' * int(pct / 2)
    print(f"   {level:20s}: {count:4d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 80)
print("CLASSIFICATION METRICS & INTERPRETATION")
print("=" * 80)

print(f"\nClassification Accuracy Distribution:")
print(f"  • All images analyzed are AI-GENERATED (from Fake folder)")
print(f"  • {ethical_count} images ({ethical_count/len(df)*100:.1f}%) have DETECTABLE artifacts")
print(f"  • {unethical_count} images ({unethical_count/len(df)*100:.1f}%) are HIGHLY CONVINCING")

print("\n" + "=" * 80)
print("ETHICAL GUIDELINES & POLICY RECOMMENDATIONS")
print("=" * 80)
print("""

🟢 ETHICAL GENERATIONS (Detectable/Safe):
   These AI-generated images contain visible artifacts that make their synthetic nature apparent:
   • Detectable artifacts protect against successful impersonation
   • Reduced risk for fraud or identity theft
   • Suitable for: Research, education, entertainment, artistic expression
   • Key Requirement: Clear disclosure that content is AI-generated
   
   Recommendation: May be used with proper labeling and consent

🔴 UNETHICAL GENERATIONS (Convincing/Risky):
   These highly realistic AI-generated images lack detectable artifacts:
   • High risk of being mistaken for authentic content
   • Potential for impersonation, fraud, deepfake attacks
   • Unsuitable for: Deception, identity theft, manipulation
   • Critical Risk: Could be used for non-consensual synthetic media
   
   Recommendation: Strict regulatory oversight required. Should require:
     1. Explicit consent from person depicted
     2. Clear technical watermarking
     3. Metadata provenance tracking
     4. Platform enforcement against misuse
     5. Legal accountability for creator

📋 BEST PRACTICES FOR RESPONSIBLE AI IMAGE GENERATION:

1. DISCLOSURE & TRANSPARENCY
   ☑ Always clearly label AI-generated content with visible markers
   ☑ Include metadata indicating generation method
   ☑ Support industry standards (C2PA Content Authenticity Initiative)
   ☑ Use digital watermarks/fingerprints

2. CONSENT & PERMISSIONS
   ☑ Obtain explicit written consent before generating images of real people
   ☑ Verify identity of consent provider
   ☑ Document consent for legal protection
   ☑ Prohibit non-consensual synthetic media creation

3. LEGAL COMPLIANCE
   ☑ Comply with deepfake, synthetic media, and identity fraud laws
   ☑ Understand jurisdiction-specific regulations
   ☑ Maintain audit logs of all creations
   ☑ Implement content moderation

4. TECHNICAL SAFEGUARDS
   ☑ Build in visible artifacts (limit convincingness)
   ☑ Embed consent information in metadata
   ☑ Use cryptographic signatures for authenticity
   ☑ Support detection and verification tools
   ☑ Implement rate limiting for mass generation

5. GOVERNANCE & ACCOUNTABILITY
   ☑ Establish ethical review boards
   ☑ Monitor for policy violations
   ☑ Report misuse to authorities
   ☑ Support victims of non-consensual synthetic media
   ☑ Regular audits and compliance checks

⚖️ LEGAL IMPLICATIONS:

Unethical AI-generated images may violate laws regarding:
   • Identity fraud and impersonation
   • Harassment and defamation
   • Deepfake regulations (increasingly common globally)
   • Non-consensual intimate imagery
   • Intellectual property rights
   • Platform terms of service
   
Creators and distributors may face:
   • Criminal prosecution
   • Civil litigation
   • Platform bans
   • Financial penalties
   • Reputation damage

📊 ANALYSIS RESULTS INTERPRETATION:

The findings show that {ec} images ({ep:.1f}%) are relatively safe due to 
detectable artifacts, while {uc} images ({up:.1f}%) pose significant ethical risks 
due to their convincing nature. This suggests:""".replace("{ec}", str(ethical_count)).replace("{ep}", f"{ethical_count/len(df)*100:.1f}").replace("{uc}", str(unethical_count)).replace("{up}", f"{unethical_count/len(df)*100:.1f}")

print("""
The findings show that""", ethical_count, """images (""", 
      f"{ethical_count/len(df)*100:.1f}", """%) are relatively safe due to 
detectable artifacts, while""", unethical_count, """images (""",
      f"{unethical_count/len(df)*100:.1f}", """%) pose significant ethical risks 
due to their convincing nature. This suggests:

1. AI technology has reached high fidelity
2. Detection becomes increasingly difficult
3. Regulatory frameworks must adapt urgently
4. Technical safeguards are critical
5. Ethical guidelines must be enforced
""".format(ethical_count=ethical_count, unethical_count=unethical_count))

print("=" * 80)
print("Sample of ETHICAL images (first 5):")
print("=" * 80)
ethical_samples = df[df['ethical_label'] == 0].head(5)
for idx, row in ethical_samples.iterrows():
    print(f"\nImage: {row['image_name']}")
    print(f"  Risk Score: {row['risk_score']:.4f}")
    print(f"  Artifact Std: {row['artifact_std']:.4f}")
    print(f"  Facial Consistency: {row['facial_consistency']:.4f}")

if unethical_count > 0:
    print("\n" + "=" * 80)
    print("Sample of UNETHICAL images (first 5):")
    print("=" * 80)
    unethical_samples = df[df['ethical_label'] == 1].head(5)
    for idx, row in unethical_samples.iterrows():
        print(f"\nImage: {row['image_name']}")
        print(f"  Risk Score: {row['risk_score']:.4f}")
        print(f"  Artifact Std: {row['artifact_std']:.4f}")
        print(f"  Facial Consistency: {row['facial_consistency']:.4f}")

print("\n" + "=" * 80)
print("Files Generated:")
print("  • ethical_classification_results.csv - Full results")
print("  • ethical_classification_visualization.png - Visual analysis charts")
print("=" * 80)
