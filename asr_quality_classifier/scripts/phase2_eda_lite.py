#!/usr/bin/env python
"""
Phase 2: Exploratory Data Analysis (EDA) - Lightweight Version

This version doesn't require external dependencies and can run on local data.
"""

import csv
from collections import Counter, defaultdict
from pathlib import Path


def analyze_csv():
    """Analyze training.csv with minimal dependencies"""
    
    print("\n" + "=" * 80)
    print("PHASE 2: EXPLORATORY DATA ANALYSIS (Lightweight)".center(80))
    print("=" * 80)
    
    # Find training.csv
    csv_path = Path(__file__).parent.parent.parent / "random" / "training.csv"
    
    if not csv_path.exists():
        print(f"\n❌ training.csv not found at: {csv_path}")
        return None
    
    print(f"\n✅ Found: {csv_path}")
    print(f"   Size: {csv_path.stat().st_size / 1024:.1f} KB")
    
    # Load and analyze
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if not data:
        print("❌ No data found in CSV")
        return None
    
    print(f"✅ Loaded {len(data)} records\n")
    
    # Basic statistics
    print("-" * 80)
    print("DATASET OVERVIEW".center(80))
    print("-" * 80)
    
    print(f"\nColumns: {', '.join(data[0].keys())}")
    print(f"Total Records: {len(data)}")
    
    # Label distribution
    print("\n" + "-" * 80)
    print("LABEL DISTRIBUTION".center(80))
    print("-" * 80 + "\n")
    
    labels = [int(row.get('label', 0)) for row in data if row.get('label', '').isdigit()]
    label_counts = Counter(labels)
    
    if not labels:
        print("❌ No valid labels found")
        return None
    
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / len(data) * 100
        label_name = "Usable ✅" if label == 1 else "Not Usable ❌" if label == 2 else f"Unknown ({label})"
        bar = "█" * int(pct / 2)
        print(f"Label {label} ({label_name:20}): {count:4d} ({pct:5.1f}%) {bar}")
    
    # Class balance
    if len(label_counts) == 2:
        counts = sorted(label_counts.values())
        ratio = counts[1] / counts[0]
        print(f"\nClass Imbalance Ratio: 1:{ratio:.2f}")
    
    # Annotator statistics
    print("\n" + "-" * 80)
    print("ANNOTATOR STATISTICS".center(80))
    print("-" * 80 + "\n")
    
    if data[0].get('username'):
        annotators = Counter(row['username'] for row in data)
        print(f"Total Annotators: {len(annotators)}\n")
        
        print("Top Annotators:")
        for user, count in annotators.most_common(5):
            pct = count / len(data) * 100
            print(f"  {user:20s}: {count:4d} ({pct:5.1f}%)")
        
        # Label distribution by annotator
        print("\n\nLabel Distribution by Top Annotators:")
        for user, _ in annotators.most_common(3):
            user_data = [row for row in data if row.get('username') == user]
            user_labels = Counter(int(row['label']) for row in user_data if row.get('label', '').isdigit())
            print(f"\n{user}:")
            for label in sorted(user_labels.keys()):
                count = user_labels[label]
                pct = count / len(user_data) * 100
                label_name = "Usable ✅" if label == 1 else "Not Usable ❌"
                print(f"  Label {label} ({label_name:15}): {count:3d} ({pct:5.1f}%)")
    
    # Folder distribution
    print("\n" + "-" * 80)
    print("FOLDER STRUCTURE".center(80))
    print("-" * 80 + "\n")
    
    if data[0].get('folder'):
        folders = Counter(row.get('folder', 'unknown') for row in data)
        print(f"Total Folders: {len(folders)}")
        print(f"Average Files per Folder: {len(data) / len(folders):.1f}\n")
        
        print("Samples per Folder:")
        for folder, count in sorted(folders.items())[:10]:
            pct = count / len(data) * 100
            print(f"  {folder:50s}: {count:3d} ({pct:5.1f}%)")
        
        if len(folders) > 10:
            print(f"  ... and {len(folders) - 10} more folders")
    
    # Transcript analysis
    print("\n" + "-" * 80)
    print("TRANSCRIPT ANALYSIS".center(80))
    print("-" * 80 + "\n")
    
    transcripts = [row.get('transcript', '') for row in data if row.get('transcript')]
    if transcripts:
        lengths = [len(t) for t in transcripts]
        word_counts = [len(t.split()) for t in transcripts]
        
        print(f"Transcripts with text: {len(transcripts)}/{len(data)}")
        print(f"\nTranscript Length (characters):")
        print(f"  Min:    {min(lengths):5d}")
        print(f"  Max:    {max(lengths):5d}")
        print(f"  Mean:   {sum(lengths)/len(lengths):5.0f}")
        print(f"  Median: {sorted(lengths)[len(lengths)//2]:5d}")
        
        print(f"\nWord Count:")
        print(f"  Min:    {min(word_counts):5d}")
        print(f"  Max:    {max(word_counts):5d}")
        print(f"  Mean:   {sum(word_counts)/len(word_counts):5.0f}")
        print(f"  Median: {sorted(word_counts)[len(word_counts)//2]:5d}")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS & RECOMMENDATIONS".center(80))
    print("=" * 80 + "\n")
    
    usable = label_counts.get(1, 0)
    not_usable = label_counts.get(2, 0)
    total = len(data)
    
    print(f"""1. CLASS BALANCE:
   ✓ Usable samples:     {usable:4d} ({usable/total*100:5.1f}%)
   ✓ Not-usable samples: {not_usable:4d} ({not_usable/total*100:5.1f}%)
   ✓ Imbalance ratio: {max(usable, not_usable)/min(usable, not_usable):.2f}:1
   → Use stratified k-fold CV and class weights in training

2. ANNOTATOR QUALITY:
   ✓ {len(annotators)} annotators contributed data
   → Check inter-annotator agreement before training

3. DATA CHARACTERISTICS:
   ✓ {len(folders)} folders with {len(data)} total samples
   ✓ Transcripts average {sum(lengths)/len(lengths):.0f} characters
   → Feature extraction should handle variable-length inputs

4. NEXT STEPS:
   ✓ Download ~4,000 samples from Azure Blob Storage
   ✓ Extract audio features (MFCC, spectrogram, SNR)
   ✓ Prepare feature matrices for modeling
   ✓ Build classifier with F1-score ≥ 0.80

""")
    
    print("=" * 80)
    print("✅ Phase 2 EDA Complete!\n")
    
    return data


if __name__ == "__main__":
    analyze_csv()
