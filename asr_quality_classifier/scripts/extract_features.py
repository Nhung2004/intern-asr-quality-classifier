#!/usr/bin/env python
"""
Phase 3: Feature Extraction & Preparation

Extracts audio, text, and cross-modal features for ML model training.
This is a comprehensive feature extraction pipeline.
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict
import math

# Try to import optional packages
try:
    import pandas as pd
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("⚠️  numpy/pandas not available - using lightweight mode")


def extract_text_features(transcript):
    """Extract basic text features from transcript"""
    if not transcript or not isinstance(transcript, str):
        return {
            'text_len_chars': 0,
            'text_len_words': 0,
            'avg_word_len': 0,
            'has_punct': 0,
            'is_empty': 1,
        }
    
    # Length features
    text_len_chars = len(transcript)
    words = transcript.split()
    text_len_words = len(words)
    
    # Average word length
    avg_word_len = text_len_chars / text_len_words if text_len_words > 0 else 0
    
    # Punctuation
    punct_count = sum(1 for c in transcript if c in '.,!?;:')
    has_punct = 1 if punct_count > 0 else 0
    
    return {
        'text_len_chars': text_len_chars,
        'text_len_words': text_len_words,
        'avg_word_len': avg_word_len,
        'punct_count': punct_count,
        'has_punct': has_punct,
        'is_empty': 0,
    }


def estimate_audio_features(file_path):
    """
    Estimate audio features from metadata.
    In real implementation, would use librosa to extract from actual audio files.
    """
    # Placeholder for when audio files are available
    return {
        'audio_exists': 0,
        'audio_duration_est': 0.0,
        'audio_has_silence': 0,
    }


def prepare_features(data, include_text=True, include_audio=False):
    """
    Prepare feature matrix for ML model
    """
    print("\n" + "=" * 80)
    print("PHASE 3: FEATURE EXTRACTION & PREPARATION".center(80))
    print("=" * 80)
    
    features_list = []
    labels = []
    file_ids = []
    
    print(f"\n📊 Processing {len(data)} samples...\n")
    
    for idx, record in enumerate(data):
        if (idx + 1) % 500 == 0:
            print(f"  ✓ Processed {idx + 1}/{len(data)} samples")
        
        # Extract label
        try:
            label = int(record.get('label', 0))
            # Convert 2 (not usable) to 0 for binary classification
            if label == 2:
                label = 0
            elif label == 1:
                label = 1
            else:
                continue  # Skip unknown labels
        except (ValueError, TypeError):
            continue
        
        # Extract file_id
        file_id = record.get('file_name', f"sample_{idx}")
        
        # Build feature vector
        features = {}
        
        # Text features
        if include_text:
            transcript = record.get('transcript', '')
            text_feats = extract_text_features(transcript)
            features.update(text_feats)
        
        # Audio features (placeholder)
        if include_audio:
            file_path = record.get('file_path', '')
            audio_feats = estimate_audio_features(file_path)
            features.update(audio_feats)
        
        # Basic metadata
        features['annotator_id'] = hash(record.get('username', 'unknown')) % 1000
        features['folder_id'] = hash(record.get('folder', 'unknown')) % 10000
        
        features_list.append(features)
        labels.append(label)
        file_ids.append(file_id)
    
    print(f"\n✅ Extracted features for {len(features_list)} samples")
    
    return features_list, labels, file_ids


def create_feature_dataframe(features_list, labels, file_ids):
    """Create pandas DataFrame from features"""
    if not HAS_NUMPY:
        return None
    
    df = pd.DataFrame(features_list)
    df['label'] = labels
    df['file_id'] = file_ids
    
    return df


def save_features(features_list, labels, file_ids, output_dir='data'):
    """Save features to JSON for model training"""
    output_path = Path(__file__).parent.parent / output_dir / 'features.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_to_save = {
        'metadata': {
            'n_samples': len(features_list),
            'n_features': len(features_list[0]) if features_list else 0,
            'feature_names': list(features_list[0].keys()) if features_list else [],
            'label_distribution': {
                'usable': sum(1 for l in labels if l == 1),
                'not_usable': sum(1 for l in labels if l == 0),
            }
        },
        'features': features_list,
        'labels': labels,
        'file_ids': file_ids,
    }
    
    with open(output_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"\n✅ Features saved to: {output_path}")
    return output_path


def analyze_features(features_list, labels):
    """Analyze extracted features"""
    if not HAS_NUMPY or not features_list:
        return
    
    print("\n" + "-" * 80)
    print("FEATURE STATISTICS".center(80))
    print("-" * 80 + "\n")
    
    df = pd.DataFrame(features_list)
    df['label'] = labels
    
    print("Feature Overview:")
    print(f"  Total Features: {len(features_list[0])}")
    print(f"  Samples: {len(features_list)}")
    
    print("\nFeature Statistics (Usable vs Not-Usable):")
    for col in df.columns:
        if col not in ['label', 'file_id'] and isinstance(df[col].iloc[0], (int, float)):
            usable_mean = df[df['label'] == 1][col].mean()
            not_usable_mean = df[df['label'] == 0][col].mean()
            print(f"\n  {col}:")
            print(f"    Usable:     {usable_mean:.2f}")
            print(f"    Not Usable: {not_usable_mean:.2f}")


def main():
    """Main feature extraction pipeline"""
    
    # Load data
    csv_path = Path(__file__).parent.parent.parent / "random" / "training.csv"
    
    if not csv_path.exists():
        print(f"\n❌ training.csv not found at: {csv_path}")
        return None
    
    print(f"\n📖 Loading data from: {csv_path}")
    
    data = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    print(f"✅ Loaded {len(data)} records")
    
    # Extract features
    features_list, labels, file_ids = prepare_features(data, include_text=True, include_audio=False)
    
    # Analyze features
    if HAS_NUMPY:
        analyze_features(features_list, labels)
    
    # Save features
    output_path = save_features(features_list, labels, file_ids)
    
    # Print summary
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY".center(80))
    print("=" * 80 + "\n")
    
    label_counts = {0: sum(1 for l in labels if l == 0), 1: sum(1 for l in labels if l == 1)}
    
    print(f"""
✅ Feature Extraction Complete!

📊 RESULTS:
   - Total Samples: {len(features_list)}
   - Usable:        {label_counts[1]} ({label_counts[1]/len(labels)*100:.1f}%)
   - Not Usable:    {label_counts[0]} ({label_counts[0]/len(labels)*100:.1f}%)
   - Features per sample: {len(features_list[0]) if features_list else 0}

💾 OUTPUT:
   - Features saved: {output_path}
   - Format: JSON with features, labels, and metadata

📈 NEXT STEPS (Phase 4):
   1. Load features from JSON
   2. Create train/val/test splits (70/15/15)
   3. Train baseline model (Logistic Regression)
   4. Evaluate on validation set
   5. Tune hyperparameters
   6. Test on held-out test set

🎯 TARGET:
   - F1-score ≥ 0.80 on test set

⚠️  NOTE:
   - Current features are lightweight (text-based only)
   - When audio files available from Azure:
     * Use librosa to extract MFCC, spectral features
     * Extract SNR, silence ratio, energy statistics
     * Use pretrained audio embeddings (Wav2Vec2)
   - Combine audio + text features for best results
    """)
    
    print("=" * 80)
    
    return features_list, labels, file_ids


if __name__ == "__main__":
    main()
