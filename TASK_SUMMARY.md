# ASR Quality Classifier - Complete Project Plan

**Status**: Comprehensive Plan ✅  
**Target Submission**: Upload to Azure Blob Storage + Email Supervisor  
**Evaluation Baseline**: F1-score ≥ 0.80

---

## 1. Overview & Actual Dataset Specification

Build a **binary classifier** to automatically classify Vietnamese **audio + transcript pairs** as:
- **Label 1 (Usable)**: Audio is clean & intelligible, transcript matches speech
- **Label 0 (Not Usable)**: Noisy audio, mismatched transcript, or poor quality

### Actual Data from Cloud (Azure Blob Storage)
| Item | Specification |
|------|---|
| **Dataset Size** | ~4,000 labeled samples |
| **Structure** | `data/audio/*.wav`, `data/transcripts/*.txt`, `data/labels.csv` |
| **Label Format** | Binary: 0 (not usable) and 1 (usable) ✅ |
| **Language** | Vietnamese |
| **Authentication** | SAS Token (env var: `AZURE_SAS_TOKEN`) |
| **Upload Output** | Personal folder (`/<username>/`) in blob |

---

## 2. Required Deliverables (5 Items)

| # | Deliverable | Details |
|---|---|---|
| 1 | **Source Code** | Python 3.9+ project, modular & documented, no hardcoded secrets |
| 2 | **Trained Model** | Serialized weights (.pt, .pkl, .onnx, etc.) |
| 3 | **Evaluation Report** | PDF/Markdown: metrics table, confusion matrix, error analysis |
| 4 | **requirements.txt** | Exact package versions for reproducibility |
| 5 | **Inference Script** | Standalone .py to load model & classify audio+transcript |

---

## 3. Technical Requirements

### System & Libraries
- **Language**: Python 3.9+
- **Cloud Integration**: `azure-storage-blob` (for data download & result upload)
- **Authentication**: Read SAS token from `AZURE_SAS_TOKEN` environment variable
- **Version Control**: Git (GitHub/GitLab private repo), commit with meaningful messages

### Code Quality Standards
- ✅ Modular architecture (separate modules for data, features, models)
- ✅ No hardcoded paths or credentials
- ✅ Comprehensive README.md (setup & run instructions)
- ✅ Type hints & docstrings where applicable
- ✅ .gitignore excludes data, models, credentials

### Reproducibility Requirements
- ✅ Exact pinned versions in requirements.txt
- ✅ Fixed random seeds for model training
- ✅ Held-out test set (no leakage during hyperparameter tuning)
- ✅ Script to retrain model from scratch using provided data

---

## 4. Comprehensive Project Phases

### **Phase 1: Setup & Cloud Data Ingestion**
**Goals**: Initialize environment, authenticate to Azure, pull data locally

**Tasks**:
- [ ] Create Python 3.9+ virtual environment
- [ ] Create `.env` file with `AZURE_SAS_TOKEN` (never commit to git)
- [ ] Write `data_loader.py`: connect to Azure Blob, download audio & transcripts
- [ ] Verify ~4,000 samples downloaded successfully
- [ ] Create project structure:
  ```
  asr_quality_classifier/
  ├── data/
  │   ├── audio/
  │   ├── transcripts/
  │   └── labels.csv
  ├── src/
  │   ├── __init__.py
  │   ├── data_loader.py       # Azure Blob download
  │   ├── feature_extractor.py # Audio/text features
  │   ├── model.py            # Model class
  │   └── utils.py            # Helpers
  ├── scripts/
  │   ├── train.py            # Training pipeline
  │   ├── evaluate.py         # Evaluation metrics
  │   └── inference.py        # Standalone classifier
  ├── notebooks/
  │   └── eda.ipynb          # Exploratory analysis
  ├── reports/
  │   └── evaluation_report.md
  ├── README.md
  ├── requirements.txt
  ├── .gitignore
  └── .env.example
  ```

**Deliverable**: `data_loader.py` + `.env.example`

---

### **Phase 2: Exploratory Data Analysis (EDA)**
**Goals**: Understand data characteristics, label distribution, data quality

**Tasks**:
- [ ] Load `labels.csv` → Check class distribution (imbalance ratio)
- [ ] Load audio files → Calculate min/max/mean duration
- [ ] Load transcripts → Calculate min/max/mean text length, language stats
- [ ] Visualize: Label distribution, audio length histogram, transcript length histogram
- [ ] Sample 5-10 pairs from each class → Listen & read for quality assessment
- [ ] Check for missing files or corrupted audio
- [ ] Compute basic statistics: vocabulary size, language diversity

**Output**: Jupyter notebook `eda.ipynb` with visualizations & findings

**Key Metrics to Report**:
- Class balance (% usable vs not usable)
- Audio duration range & distribution
- Transcript length range & distribution
- Any obvious data quality issues

---

### **Phase 3: Feature Extraction**
**Goals**: Convert raw audio & text into machine-learning features

#### **3.1 Audio Features**
Extract from each `.wav` file:
- **MFCCs** (Mel-Frequency Cepstral Coefficients): 13-20 coefficients, capture mel-scale spectrum
- **Log-Mel Spectrogram**: Time-frequency energy distribution
- **Energy Statistics**: Mean/std/min/max energy per frame
- **Pitch Estimation**: Fundamental frequency (F0) via autocorrelation or PYIN
- **Silence Ratio**: % of frames below energy threshold
- **Signal-to-Noise Ratio (SNR)**: Estimate via noise floor detection
- **Pretrained Embeddings** (optional): Wav2Vec2 or Whisper encoder output (768-dim vectors)

**Libraries**: `librosa`, `scipy`, `numpy`, `torch` (for embeddings)

#### **3.2 Transcript Features**
Extract from each `.txt` file:
- **Length**: Character count, word count, token count
- **Vocabulary Metrics**: 
  - Unique words / total words
  - Out-Of-Vocabulary (OOV) rate (vs common Vietnamese dictionary)
- **Language Model Perplexity**: Via pretrained Vietnamese LM (if available)
- **Embeddings**: Sentence-transformer (e.g., `distiluse-base-multilingual-cased-v2`) → 768-dim vector

**Libraries**: `transformers`, `underthesea` (Vietnamese NLP)

#### **3.3 Cross-Modal Features** (Optional but Recommended)
- **Transcript-Audio Alignment**: Run ASR on audio, compare hypothesis vs ground-truth
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Cosine similarity of embeddings
- **Linguistic Consistency**: Grammar/syntax checks on transcript

**Output**: `feature_extractor.py` module + feature matrices saved as `.npz` or `.pkl`

---

### **Phase 4: Model Selection & Training**
**Goals**: Build classifier, optimize hyperparameters, achieve F1 ≥ 0.80

#### **4.1 Baseline Models (Start Here)**
1. **Logistic Regression** on hand-crafted features
2. **Gradient Boosting** (XGBoost, LightGBM) on hand-crafted features
3. **Random Forest** on combined features

#### **4.2 Intermediate Models**
4. **Neural Network** (shallow MLP): Input hand-crafted features
5. **Audio-specific NN**: CNN on spectrograms
6. **Multimodal NN**: Concatenate audio + text embeddings → dense layers

#### **4.3 Advanced Models (if time permits)**
7. **Transformer-based**: Fine-tune a pretrained model on audio+text
8. **Ensemble**: Combine best baseline + deep models

#### **4.4 Training Strategy**
- **Data Split**: 
  - Train: 70% (~2,800 samples)
  - Validation: 15% (~600 samples)
  - Test: 15% (~600 samples)
  - **Use stratified split** to preserve class distribution

- **Evaluation Protocol**:
  - Train on train set
  - Tune hyperparameters on validation set (never on test!)
  - Report final metrics on test set only
  
- **Class Imbalance Handling**:
  - Use `class_weight='balanced'` in sklearn models
  - Or oversample minority class / undersample majority class
  - Use F1-score (macro) as primary metric

- **Cross-Validation** (optional):
  - Stratified 5-fold CV to estimate generalization

**Output**: `train.py` script + trained model file (`.pkl`, `.pt`, `.h5`, etc.)

---

### **Phase 5: Evaluation & Metrics**
**Goals**: Rigorous evaluation on held-out test set

**Metrics to Report**:
| Metric | Formula / Interpretation |
|--------|---|
| **Precision** | TP / (TP + FP) → % of predicted usable that are actually usable |
| **Recall** | TP / (TP + FN) → % of actual usable samples caught |
| **F1-Score (Macro)** | 2·(Precision·Recall)/(Precision+Recall) → **PRIMARY METRIC** |
| **ROC-AUC** | Area under ROC curve → probability ranking quality |
| **Confusion Matrix** | TN, FP, FN, TP breakdown |
| **Per-Class Metrics** | Separate F1 for usable & not-usable |

**Output**: `evaluate.py` script generates:
- Metrics table (CSV/Markdown)
- Confusion matrix plot
- ROC curve plot
- Summary JSON

**Target**: F1-score ≥ 0.80 on test set

---

### **Phase 6: Error Analysis & Refactoring**
**Goals**: Understand failure modes, improve code quality, ensure reproducibility

#### **6.1 Error Analysis**
- [ ] Identify false positives (predicted usable, actually not)
- [ ] Identify false negatives (predicted not usable, actually usable)
- [ ] Sample 20-30 misclassified examples
- [ ] Categorize errors:
  - Audio quality issues (noise, low volume)
  - Transcript mismatches (typos, wrong words)
  - Edge cases (very short/long samples)
- [ ] Document patterns & recommendations

#### **6.2 Code Refactoring**
- [ ] Modularize feature extraction
- [ ] Add comprehensive docstrings & type hints
- [ ] Create `utils.py` for common operations
- [ ] Add CLI args to `train.py` & `inference.py`
- [ ] Write `.gitignore` (exclude `/data`, `/models`, `.env`)

#### **6.3 Reproducibility Check**
- [ ] Test: Delete trained model, retrain from scratch → Verify same metrics
- [ ] Document exact Python version & OS used
- [ ] Pin all package versions in `requirements.txt`
- [ ] Write detailed README.md:
  - System requirements
  - Installation steps
  - How to set `AZURE_SAS_TOKEN`
  - How to run training, evaluation, inference
  - Expected outputs

**Output**: Clean, modular codebase + reproducible training script

---

### **Phase 7: Reporting**
**Goals**: Communicate findings & methodology clearly

#### **Evaluation Report Contents** (Markdown or PDF)
1. **Executive Summary** (1 paragraph)
   - Problem, approach, key result
   
2. **Dataset Overview**
   - ~4,000 samples, class distribution (% usable vs not)
   - Audio duration range, transcript length stats
   
3. **Methodology**
   - Features used (audio: MFCCs, transcripts: embeddings, etc.)
   - Model architecture & hyperparameters
   - Train/val/test split strategy
   - Class imbalance handling
   
4. **Results**
   - Metrics table: Precision, Recall, F1, ROC-AUC
   - Confusion matrix (visual)
   - Per-class breakdown
   
5. **Error Analysis**
   - Top error categories (noisy audio, transcription mismatches, etc.)
   - Example misclassifications (2-3 concrete samples)
   - Root cause analysis
   
6. **Lessons Learned**
   - What features were most predictive?
   - What challenges emerged?
   - Recommendations for improvement

**Output**: `evaluation_report.md` (or `.pdf`)

---

### **Phase 8: Submission & Communication**
**Goals**: Deliver all artifacts to supervisor & Azure Blob

#### **8.1 Final Deliverables Package**
Create folder: `/<your_name>/` on Azure Blob with:
```
/<your_name>/
├── source_code/              # Zipped Python project
│   └── asr_quality_classifier.zip
├── trained_model/            # Model file(s)
│   ├── model.pkl
│   └── model_metadata.json
├── evaluation_report.md      # Main report
├── requirements.txt          # Exact dependencies
├── inference.py             # Standalone classifier
├── README.md                # Setup instructions
└── eda_summary.json         # EDA findings
```

#### **8.2 Email to Supervisor**
**Subject**: ASR Quality Classifier - Submission Complete

**Body** (3 required sections):
1. ✅ **Confirmation**: "I have successfully uploaded all deliverables to `/<my_name>/` in the Azure Blob Storage."

2. 📊 **Best Metrics**: 
   - F1-Score (Macro): [X.XX]
   - ROC-AUC: [X.XX]
   - (on validation set during development, final test set in report)

3. 📝 **One-Paragraph Summary**:
   - Describe your approach in plain language
   - Example: "I extracted MFCCs and spectrograms from audio, combined with embeddings from the transcript using sentence-transformers, then trained an XGBoost classifier. The model achieved F1=0.82 on the test set by effectively identifying noisy audio (SNR<10dB) and transcript mismatches (WER>30%). The main challenge was Vietnamese text preprocessing; I used the `underthesea` library for tokenization..."

---

## 5. Detailed Timeline & Milestones

| Phase | Est. Duration | Key Milestones |
|-------|---|---|
| 1: Setup & Ingestion | 2-3 hours | ✅ Data downloaded, project structure ready |
| 2: EDA | 4-6 hours | ✅ Dataset characteristics understood, visualizations done |
| 3: Feature Extraction | 6-8 hours | ✅ Feature matrices generated & validated |
| 4: Modeling | 8-12 hours | ✅ Baseline & deep models trained, hyperparams tuned |
| 5: Evaluation | 2-3 hours | ✅ Metrics computed, test set results locked |
| 6: Error Analysis & Refactor | 4-6 hours | ✅ Code clean, reproducibility verified |
| 7: Reporting | 3-4 hours | ✅ Report written with findings |
| 8: Submission | 1-2 hours | ✅ Upload complete, email sent |
| **TOTAL** | **~30-45 hours** | — |

---

## 6. Evaluation Criteria & Scoring

| Criterion | Weight | Target |
|-----------|--------|--------|
| **Model Performance** (F1, ROC-AUC on test set) | 40% | F1 ≥ 0.80, ROC-AUC ≥ 0.85 |
| **Code Quality** (modularity, documentation, no secrets) | 25% | Well-organized, readable, commented |
| **Reproducibility** (can retrain from scratch) | 20% | requirements.txt pinned, seed fixed, script runs end-to-end |
| **Report & Analysis** (honest error analysis, lessons) | 15% | Clear findings, concrete examples, improvements noted |

**Total Score = 40% × Performance + 25% × Quality + 20% × Reproducibility + 15% × Reporting**

---

## 7. Important Constraints & Best Practices

### ✅ Do's
- Commit code to git frequently with meaningful messages
- Use environment variable for `AZURE_SAS_TOKEN`
- Save model + metadata for reproducibility
- Test entire pipeline end-to-end before submission
- Document assumptions & design choices
- Focus on minority class (not-usable) performance

### ❌ Don'ts
- Use test set for hyperparameter tuning (data leakage!)
- Hardcode file paths or credentials in code
- Retrain on full dataset without held-out test set
- Skip error analysis or honest reporting
- Forget to pin package versions
- Submit without testing reproducibility

---

## 8. Next Steps (Immediate Actions)

1. **Wait for SAS token** from supervisor
2. **Set up Python 3.9+ environment** with `azure-storage-blob`
3. **Start Phase 1**: Write `data_loader.py` to authenticate & download data
4. **Begin Phase 2**: EDA notebook to understand ~4,000 samples
5. **Proceed sequentially** through phases 3-8

---

**Good luck! Target submission: F1-score ≥ 0.80, full reproducibility, clean code.** 🚀
