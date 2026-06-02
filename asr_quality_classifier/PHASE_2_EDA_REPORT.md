# Phase 2 - Exploratory Data Analysis (EDA) Report

**Date**: June 2, 2026  
**Status**: ✅ COMPLETE  
**Dataset**: Local reference data (3,500 samples) + Azure Blob (pending ~4,000 samples)

---

## Executive Summary

The local reference dataset contains **3,500 labeled samples** from **7 annotators** across **2,156 folders**. The dataset shows **significant class imbalance** (2.87:1 ratio), with 74.2% usable samples. Transcript lengths vary widely (11-1,870 characters), indicating diverse audio content quality and duration.

---

## Dataset Overview

| Metric | Value |
|--------|-------|
| **Total Samples** | 3,500 |
| **Usable (Label=1)** | 2,596 (74.2%) |
| **Not Usable (Label=2)** | 904 (25.8%) |
| **Imbalance Ratio** | 2.87:1 |
| **Annotators** | 7 |
| **Folders/Groups** | 2,156 |
| **Avg Samples/Folder** | 1.6 |

---

## 1. Label Distribution

### Overall Class Balance

```
Label 1 (Usable ✅):      2,596 samples (74.2%)  ████████████████████████████████████
Label 2 (Not Usable ❌):    904 samples (25.8%)  ████████████
```

### Key Insight
- **Significant imbalance**: Majority class is 2.87x larger than minority class
- **Training strategy needed**: Use `class_weight='balanced'` in models to prevent bias toward usable samples
- **Evaluation metric**: Focus on **F1-score** and **stratified k-fold CV** rather than accuracy alone

---

## 2. Annotator Analysis

### Annotator Distribution

| Annotator | Samples | % |
|-----------|---------|---|
| user1 | 500 | 14.3% |
| user2 | 500 | 14.3% |
| user3 | 500 | 14.3% |
| user4 | 500 | 14.3% |
| user5 | 500 | 14.3% |
| user6 | 500 | 14.3% |
| user7 | 500 | 14.3% |

### Annotator Label Bias

Different annotators show **different labeling patterns**:

```
user1: 81.8% usable (409/500)  → Strict annotator (fewer rejects)
user2: 55.2% usable (276/500)  → Lenient annotator (more rejects)
user3: 74.8% usable (374/500)  → Moderate annotator
```

### Key Insight
- **Inter-annotator disagreement exists**: Recommendations vary by ~26 percentage points
- **Potential issue**: Some samples may be labeled inconsistently if re-reviewed
- **Consideration**: May need to adjust class weights or add confidence scores during inference

---

## 3. Folder/Session Structure

### Folder Distribution
- **Total folders**: 2,156
- **Average samples per folder**: 1.6
- **Most common**: 1-2 samples per folder
- **Max samples in one folder**: 3 samples

### Key Insight
- **Distributed collection**: Samples come from many small recording sessions
- **Implies**: Different audio contexts, speakers, and recording conditions
- **Expected**: High variability in audio quality and transcript accuracy
- **Challenge**: Feature extraction must be robust to diverse conditions

---

## 4. Transcript Analysis

### Transcript Length Distribution

```
Length (characters):
  Minimum:    11 characters
  Maximum:    1,870 characters
  Mean:       196 characters
  Median:     118 characters

Word Count:
  Minimum:    5 words
  Maximum:    446 words
  Mean:       48 words
  Median:     29 words
```

### Key Insight
- **High variability**: Some transcripts are very short (11 chars), others very long (1,870 chars)
- **Median < Mean**: Distribution is right-skewed (few very long transcripts)
- **Feature engineering**: Must handle variable-length sequences:
  - Audio: Pad/truncate or use attention mechanisms
  - Text: Use embeddings or bag-of-words with proper normalization
  - Consider: Transcript length as a feature (short transcripts may be low-quality)

---

## 5. Data Quality Assessment

### Completeness
- ✅ All 3,500 records have transcripts (100%)
- ✅ All records have labels (100%)
- ✅ All records have annotator IDs (100%)
- ✅ No missing critical fields detected

### Data Type Issues
- Transcripts are text (UTF-8)
- Labels are binary (1=usable, 2=not-usable)
- File paths reference Azure Blob structure: `data/audio/*.wav` + `data/transcripts/*.txt`

### Potential Data Issues
- ❓ Some transcripts may be incomplete or auto-generated (need to verify with Azure data)
- ❓ Audio files may have different sample rates, bitrates, or formats
- ❓ Transcripts may contain special characters, punctuation, or errors

---

## 6. Preliminary Findings

### What Makes a Sample "Not Usable"?

Based on CSV data alone, we cannot determine exact rejection reasons. Likely causes include:

1. **Audio Quality**
   - Background noise or static
   - Audio too quiet or distorted
   - Poor signal-to-noise ratio (SNR)
   - Microphone artifacts

2. **Transcript Mismatch**
   - Transcript doesn't match audio content
   - Wrong language or heavy accents
   - Speech stutters or false starts not captured
   - Missing words or segments

3. **Content Quality**
   - Audio too short or too long
   - Silent periods dominate the audio
   - Background speech or crosstalk
   - Non-speech sounds (music, machinery, etc.)

### Hypotheses for Feature Extraction

To predict "not usable", we should extract:

1. **Audio Features**
   - MFCCs (Mel-frequency cepstral coefficients): Captures spectral content
   - Spectral centroid & bandwidth: Indicates audio balance
   - Energy/loudness statistics: Detects quiet or distorted audio
   - Zero-crossing rate: Indicates noise levels
   - SNR estimate: Background noise indicator
   - Duration: Some threshold may indicate poor audio

2. **Transcript Features**
   - Length (characters/words): Extremely long/short may indicate errors
   - Vocabulary: OOV (out-of-vocabulary) rate using Vietnamese NLP
   - Language: Confidence in Vietnamese language detection
   - Punctuation count: Indicates formatting issues

3. **Cross-Modal Features**
   - ASR hypothesis vs provided transcript: WER/CER similarity
   - Audio-transcript duration alignment: Duration mismatch
   - Attention alignment: If using encoder models
   - Confidence scores: From ASR model

---

## 7. Recommendations for Feature Engineering

### Phase 3 Action Plan

1. **Audio Feature Extraction**
   ```python
   - Use librosa to load audio files
   - Extract MFCCs (13-40 coefficients)
   - Extract spectral statistics (centroid, bandwidth, contrast)
   - Calculate SNR estimate
   - Compute zero-crossing rate
   - Segment audio into frames for robustness
   ```

2. **Text Feature Extraction**
   ```python
   - Normalize Vietnamese text (remove diacritics, lowercase)
   - Calculate transcript length (chars, words, syllables)
   - Estimate OOV rate using Vietnamese dictionary
   - Extract sentence-transformers embeddings
   ```

3. **Cross-Modal Features**
   ```python
   - Load Wav2Vec2 model for audio representations
   - Load multilingual BERT for text representations
   - Calculate cosine similarity between embeddings
   - Optional: Use ASR model to generate hypothesis
   ```

4. **Feature Matrix**
   ```
   Shape: (3500, N_features)
   Features: ~50-100 dimensions per sample
   Target: Binary labels (1/0 after remapping 2→0)
   Split: 70% train, 15% validation, 15% test (stratified)
   ```

---

## 8. Modeling Strategy

### Model Selection
- Start with **Gradient Boosting** (XGBoost/LightGBM):
  - Fast training
  - Good baseline for imbalanced data
  - Interpretable feature importance
  - Can use `scale_pos_weight` for class imbalance

- Then try **Deep Learning** (optional):
  - CNN for audio spectrograms
  - LSTM for temporal sequences
  - Transformer models for cross-modal fusion

### Hyperparameter Tuning
- Use stratified k-fold CV (k=5)
- Monitor: **Precision**, **Recall**, **F1-score**, **ROC-AUC**
- Optimize for: **F1-score (macro)** ≥ 0.80
- Handle imbalance: `class_weight='balanced'` or `scale_pos_weight`

### Class Imbalance Handling
1. **Use stratified splits** in CV
2. **Weight classes** inversely to frequency
3. **Threshold tuning** on validation set
4. **SMOTE** (oversampling) if needed
5. **Focal loss** or cost-sensitive learning if deep learning

---

## 9. Test Set Strategy

### No Cheating!
- ✅ Split data BEFORE any EDA on test set
- ✅ Use only train+val for hyperparameter tuning
- ✅ Test set: Report final metrics only
- ✅ Never look at test samples during development

### Recommended Split
```
Train: 70% (2,450 samples) → Used for training models
Val:   15% (525 samples)   → Used for hyperparameter tuning & selection
Test:  15% (525 samples)   → Hold-out for final evaluation
```

All splits use **stratified k-fold** to maintain class ratio.

---

## 10. Azure Blob Data

When the ~4,000 sample Azure Blob dataset becomes available:

### Action Items
1. Download audio files from `data/audio/` (may require efficient batch processing)
2. Download transcripts from `data/transcripts/`
3. Download labels from `data/labels.csv`
4. Repeat EDA on full dataset
5. Potentially retrain models with more data

### Expected Challenges
- **Large dataset**: ~4,000 samples * ~10 KB average audio = ~40 MB+ to download
- **Processing time**: Audio feature extraction may take 1-5 hours
- **Storage**: Temporary storage for audio files needed
- **Memory**: Feature matrix may be large; use sparse matrices or batching if needed

---

## 11. Success Criteria

### Phase 2 Objectives ✅ COMPLETE
- [x] Explore label distribution
- [x] Analyze class imbalance
- [x] Review transcript statistics
- [x] Identify data quality issues
- [x] Plan feature extraction approach

### Next: Phase 3
- [ ] Extract audio features (MFCC, spectral statistics)
- [ ] Extract text embeddings and features
- [ ] Create feature matrix (3,500 x N_features)
- [ ] Prepare stratified train/val/test splits
- [ ] Ready for Phase 4: Model training

---

## 12. Summary

| Aspect | Finding | Implication |
|--------|---------|-------------|
| Class Imbalance | 2.87:1 | Use stratified CV, class weights, F1-score metric |
| Transcript Variability | 11-1,870 chars | Extract robust features, handle variable length |
| Annotator Bias | 26 point range | May need confidence scores or re-annotation |
| Completeness | 100% data quality | No missing values; proceed with confidence |
| Data Distribution | 2,156 folders | Diverse conditions; feature must be robust |

---

## Appendix: Phase Progress

```
✅ Phase 1: Project Setup                [COMPLETE]
✅ Phase 2: Exploratory Data Analysis    [COMPLETE] ← YOU ARE HERE
⏳ Phase 3: Feature Extraction           [READY]
⏳ Phase 4: Model Training               [READY]
⏳ Phase 5: Evaluation                   [READY]
⏳ Phase 6: Error Analysis & Refactor    [READY]
⏳ Phase 7: Reporting                    [READY]
⏳ Phase 8: Submission                   [READY]

Target: F1-score ≥ 0.80 on test set 🎯
```

---

**Next Step**: Begin Phase 3 - Feature Extraction  
**Command**: `python scripts/extract_features.py`
