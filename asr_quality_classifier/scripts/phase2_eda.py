#!/usr/bin/env python
"""
Phase 2: Exploratory Data Analysis (EDA)

Analyze the local training.csv data to understand:
- Label distribution (usable vs not-usable)
- Transcript characteristics
- Data quality
- Class imbalance
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.utils import setup_logger, set_random_seed, print_config

logger = setup_logger(__name__, log_level=Config.LOG_LEVEL)


def analyze_local_data():
    """Analyze training.csv data available locally"""
    
    logger.info("=" * 80)
    logger.info("PHASE 2: EXPLORATORY DATA ANALYSIS".center(80))
    logger.info("=" * 80)
    
    # Check if training.csv exists
    training_csv = Path(__file__).parent.parent.parent / "random" / "training.csv"
    
    if not training_csv.exists():
        logger.error(f"❌ training.csv not found at: {training_csv}")
        logger.info("Will wait for data download from Azure Blob...")
        return None
    
    logger.info(f"✅ Found training.csv: {training_csv}")
    logger.info(f"   File size: {training_csv.stat().st_size / 1024:.1f} KB")
    
    # Load data
    logger.info("\n📖 Loading data...")
    df = pd.read_csv(training_csv)
    logger.info(f"✅ Loaded {len(df)} records")
    
    # Display basic info
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW".center(80))
    print("=" * 80)
    print(f"Total Records: {len(df)}")
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print()
    
    # Label distribution
    print("-" * 80)
    print("LABEL DISTRIBUTION".center(80))
    print("-" * 80)
    
    # Note: CSV uses label values 1 (usable) and 2 (unusable), not 0/1
    label_dist = df['label'].value_counts().sort_index()
    print(f"\nLabel Value Counts:")
    for label_val, count in label_dist.items():
        pct = count / len(df) * 100
        label_name = "Usable ✅" if label_val == 1 else "Not Usable ❌" if label_val == 2 else "Unknown"
        print(f"  Label {label_val} ({label_name:15}): {count:4d} ({pct:5.1f}%)")
    
    # Class balance
    if len(label_dist) == 2:
        counts = label_dist.values
        imbalance_ratio = max(counts) / min(counts)
        print(f"\nClass Imbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"Minority Class: {label_dist.idxmin()} ({label_dist.min()} samples)")
        print(f"Majority Class: {label_dist.idxmax()} ({label_dist.max()} samples)")
    
    # Data quality
    print("\n" + "-" * 80)
    print("DATA QUALITY".center(80))
    print("-" * 80)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✅ No missing values")
    else:
        for col in df.columns:
            if missing[col] > 0:
                print(f"  ❌ {col}: {missing[col]} missing")
    
    # Transcript analysis
    print("\n" + "-" * 80)
    print("TRANSCRIPT ANALYSIS".center(80))
    print("-" * 80)
    
    df['transcript_len'] = df['transcript'].fillna('').str.len()
    df['transcript_words'] = df['transcript'].fillna('').str.split().str.len()
    
    print(f"\nTranscript Length (characters):")
    print(f"  Mean: {df['transcript_len'].mean():.0f}")
    print(f"  Min:  {df['transcript_len'].min():.0f}")
    print(f"  Max:  {df['transcript_len'].max():.0f}")
    print(f"  Std:  {df['transcript_len'].std():.0f}")
    
    print(f"\nTranscript Length (words):")
    print(f"  Mean: {df['transcript_words'].mean():.0f}")
    print(f"  Min:  {df['transcript_words'].min():.0f}")
    print(f"  Max:  {df['transcript_words'].max():.0f}")
    print(f"  Std:  {df['transcript_words'].std():.0f}")
    
    # Annotator distribution
    print("\n" + "-" * 80)
    print("ANNOTATOR DISTRIBUTION".center(80))
    print("-" * 80)
    
    if 'username' in df.columns:
        annotators = df['username'].value_counts()
        print(f"\nTotal Annotators: {len(annotators)}")
        print(f"\nTop 5 Annotators:")
        for annotator, count in annotators.head(5).items():
            pct = count / len(df) * 100
            print(f"  {annotator:10s}: {count:4d} ({pct:5.1f}%)")
    
    # File structure analysis
    print("\n" + "-" * 80)
    print("FILE STRUCTURE".center(80))
    print("-" * 80)
    
    if 'folder' in df.columns:
        unique_folders = df['folder'].nunique()
        print(f"\nUnique Folders: {unique_folders}")
        print(f"Avg Files per Folder: {len(df) / unique_folders:.1f}")
    
    # Label by annotator (sample)
    print("\n" + "-" * 80)
    print("LABEL DISTRIBUTION BY ANNOTATOR (Top 3)".center(80))
    print("-" * 80)
    
    if 'username' in df.columns and 'label' in df.columns:
        for user in df['username'].value_counts().head(3).index:
            user_data = df[df['username'] == user]
            label_dist_user = user_data['label'].value_counts().sort_index()
            print(f"\n{user}:")
            for label_val, count in label_dist_user.items():
                pct = count / len(user_data) * 100
                label_name = "Usable ✅" if label_val == 1 else "Not Usable ❌"
                print(f"  Label {label_val} ({label_name:15}): {count:3d} ({pct:5.1f}%)")
    
    # Key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS & RECOMMENDATIONS".center(80))
    print("=" * 80)
    
    usable_count = (df['label'] == 1).sum()
    not_usable_count = (df['label'] == 2).sum()
    total = len(df)
    
    print(f"""
1. CLASS BALANCE:
   - {usable_count/total*100:.1f}% samples are usable
   - {not_usable_count/total*100:.1f}% samples are not usable
   - Imbalance ratio: {max(usable_count, not_usable_count) / min(usable_count, not_usable_count):.2f}:1
   - Recommendation: Use stratified k-fold CV, balance class weights in training

2. TRANSCRIPT CHARACTERISTICS:
   - Average length: {df['transcript_len'].mean():.0f} characters
   - Range: {df['transcript_len'].min():.0f} - {df['transcript_len'].max():.0f}
   - Some transcripts may be too short/long (outliers to investigate)

3. DATA QUALITY:
   - {annotators.nunique() if 'username' in df.columns else 'N/A'} different annotators
   - Potential inter-annotator agreement issues
   - Recommendation: Check for annotation consistency

4. NEXT STEPS FOR PHASE 3:
   - Extract audio features (MFCCs, spectrograms, SNR)
   - Extract text embeddings from transcripts
   - Cross-modal features (ASR alignment)
   - Prepare feature matrices for modeling
    """)
    
    print("=" * 80 + "\n")
    
    return df


def main():
    """Main EDA function"""
    set_random_seed(Config.RANDOM_SEED)
    
    logger.info("🚀 Starting Phase 2 - EDA")
    logger.info(f"Random Seed: {Config.RANDOM_SEED}")
    
    df = analyze_local_data()
    
    if df is not None:
        logger.info("✅ Phase 2 EDA Complete!")
        logger.info("\n📊 Next: Proceed to Phase 3 - Feature Extraction")
        logger.info("   Run: python scripts/extract_features.py")
    else:
        logger.info("⏳ Waiting for Azure Blob data download")
        logger.info("   Run: python scripts/download_data.py (after setting AZURE_SAS_TOKEN)")


if __name__ == "__main__":
    main()
