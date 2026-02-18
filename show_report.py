import pandas as pd

df = pd.read_csv('ethical_classification_results.csv')

ethical_count = len(df[df['ethical_label'] == 0])
unethical_count = len(df[df['ethical_label'] == 1])
total = len(df)

print("=" * 80)
print("FHIBE ETHICAL CLASSIFICATION REPORT - AI-GENERATED IMAGES")
print("=" * 80)

print(f"\n📊 DATASET SUMMARY:")
print(f"   Total Images Analyzed: {total}")
print(f"   ✓ ETHICAL (Detectable/Safe): {ethical_count} ({100*ethical_count/total:.1f}%)")
print(f"   ✗ UNETHICAL (Convincing/Risky): {unethical_count} ({100*unethical_count/total:.1f}%)")

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
    pct = 100*count / total
    bar = '█' * int(pct / 2)
    print(f"   {level:20s}: {count:4d} ({pct:5.1f}%) {bar}")

print("\n" + "=" * 80)
print("CLASSIFICATION KEY FINDINGS")
print("=" * 80)

print(f"\n✓ KEY RESULT: {ethical_count/total*100:.1f}% of AI-generated images are ETHICALLY classified")
print(f"  (have detectable artifacts, lower misuse risk)")
print(f"\n✗ KEY CONCERN: {unethical_count/total*100:.1f}% are HIGH RISK")
print(f"  (convincing with minimal detectable artifacts)")

print("\n" + "=" * 80)
print("ETHICAL GUIDELINES & RECOMMENDATIONS")
print("=" * 80)

print("""
🟢 ETHICAL GENERATIONS (Detectable/Safe):
   • Visible artifacts make synthetic nature apparent
   • Lower risk of successful impersonation
   • Suitable for: Research, education, entertainment, art
   • Requirement: Clear AI-generation disclosure
   
   ✓ Recommendation: May be used with proper labeling

🔴 UNETHICAL GENERATIONS (Convincing/Risky):
   • Highly realistic with minimal detectable artifacts
   • High misuse risk for fraud, impersonation
   • Unsuitable for: Any deceptive purpose
   • Critical Risk: Non-consensual synthetic media
   
   ✓ Requirement: Strict regulations + explicit consent

📋 BEST PRACTICES:

1. DISCLOSURE
   ☑ Clearly label all AI-generated content
   ☑ Include generation method in metadata
   ☑ Support C2PA/content authenticity standards

2. CONSENT
   ☑ Explicit consent before generating images of real people
   ☑ Verify consent provider identity
   ☑ Document consent for legal protection

3. LEGAL COMPLIANCE
   ☑ Comply with local deepfake/synthetic media laws
   ☑ Maintain creation audit logs
   ☑ Implement content moderation

4. TECHNICAL SAFEGUARDS
   ☑ Build in detectable artifacts where possible
   ☑ Embed metadata and digital signatures
   ☑ Use watermarks for identification
   ☑ Support detection tools

5. ACCOUNTABILITY
   ☑ Establish ethical review processes
   ☑ Monitor for policy violations
   ☑ Report misuse to authorities
   ☑ Support victims of harmful content

⚖️ LEGAL IMPLICATIONS:

Misuse of AI-generated images may violate laws regarding:
   • Identity fraud and impersonation
   • Deepfake regulations
   • Non-consensual intimate imagery
   • Harassment and defamation
   • Platform terms of service

Consequences may include:
   • Criminal prosecution
   • Civil litigation
   • Platform bans
   • Financial penalties
""")

print("=" * 80)
print("Sample ETHICAL Images:")
print("=" * 80)
ethical_samples = df[df['ethical_label'] == 0].head(5)
for idx, row in ethical_samples.iterrows():
    print(row['image_name'] + ": Risk=" + f"{row['risk_score']:.4f}" + ", Artifacts=" + f"{row['artifact_std']:.2f}")

if unethical_count > 0:
    print("\n" + "=" * 80)
    print("Sample UNETHICAL Images:")
    print("=" * 80)
    unethical_samples = df[df['ethical_label'] == 1].head(5)
    for idx, row in unethical_samples.iterrows():
        print(f"{row['image_name']}: Risk={row['risk_score']:.4f}, Artifacts={row['artifact_std']:.2f}")

print("\n" + "=" * 80)
print("FILES GENERATED:")
print("=" * 80)
print("  ✓ ethical_classification_results.csv - Full analysis data")
print("  ✓ ethical_classification_visualization.png - Charts and graphs")
print("=" * 80)
