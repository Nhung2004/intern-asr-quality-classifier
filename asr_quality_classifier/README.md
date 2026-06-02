# ASR Quality Classifier

Automated classifier to identify usable vs non-usable audio-transcript pairs for ASR (Automatic Speech Recognition) training.

**Target**: F1-score ≥ 0.80 on held-out test set

---

## 📋 Overview

This project builds a machine learning system to automatically classify Vietnamese audio recordings and their transcripts as:
- **Label 1 (Usable)**: Clean audio, correct transcript match
- **Label 0 (Not Usable)**: Noisy audio, transcript mismatch, poor quality

### Dataset
- **Size**: ~4,000 labeled samples
- **Source**: Azure Blob Storage
- **Language**: Vietnamese
- **Format**: WAV audio + TXT transcripts + CSV labels

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.9+
- Git
- Windows/Linux/macOS

### 2. Setup Environment

#### Clone Repository
```bash
cd c:\Users\AD\random
# (Repository already exists as asr_quality_classifier/)
cd asr_quality_classifier
```

#### Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Azure Credentials

Create a `.env` file in project root:
```bash
cp .env.example .env
```

Edit `.env` and add your Azure SAS token:
```
AZURE_SAS_TOKEN=your_sas_token_from_supervisor
```

⚠️ **Never commit `.env` to git** - it contains credentials!

### 4. Download Data from Azure Blob

```bash
python scripts/download_data.py
```

This will download ~4,000 audio files, transcripts, and labels to `./data/`

---

## 📁 Project Structure

```
asr_quality_classifier/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── config.py                # Configuration & constants
│   ├── data_loader.py           # Azure Blob download & data loading
│   ├── feature_extractor.py     # Audio/text feature extraction
│   ├── model.py                 # Model classes
│   └── utils.py                 # Utility functions
├── scripts/                      # Executable scripts
│   ├── download_data.py         # Download from Azure Blob
│   ├── train.py                 # Model training pipeline
│   ├── evaluate.py              # Evaluation on test set
│   └── inference.py             # Standalone classifier
├── notebooks/
│   └── eda.ipynb                # Exploratory Data Analysis
├── data/                        # Data (git-ignored)
│   ├── audio/
│   ├── transcripts/
│   └── labels.csv
├── models/                      # Trained models (git-ignored)
├── reports/                     # Evaluation reports
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

---

## 🔄 Project Phases

### Phase 1: Setup & Data Ingestion ✅
- [x] Create project structure
- [ ] Download data from Azure Blob
- [ ] Verify ~4,000 samples

### Phase 2: EDA (Exploratory Data Analysis)
```bash
jupyter notebook notebooks/eda.ipynb
```
- Analyze label distribution
- Audio duration & transcript statistics
- Identify data quality issues

### Phase 3: Feature Extraction
```bash
python scripts/extract_features.py
```
**Audio Features**: MFCCs, log-Mel spectrograms, energy stats, SNR, silence ratio  
**Transcript Features**: Length, embeddings, language stats  
**Cross-modal**: WER/CER similarity

### Phase 4: Model Training
```bash
python scripts/train.py \
  --model xgboost \
  --epochs 100 \
  --batch_size 32
```

Supported models:
- `logistic_regression`
- `random_forest`
- `xgboost`
- `lightgbm`
- `neural_network` (MLP)
- `cnn_audio` (CNN on spectrograms)
- `multimodal` (Audio + Text embeddings)

### Phase 5: Evaluation
```bash
python scripts/evaluate.py --model models/best_model.pkl
```

Outputs: Metrics, confusion matrix, ROC curve

### Phase 6: Inference
```bash
python scripts/inference.py \
  --audio path/to/audio.wav \
  --transcript "sample transcript" \
  --model models/best_model.pkl
```

Returns: `{"prediction": 1, "confidence": 0.92}`

---

## 📊 Expected Results

| Metric | Target |
|--------|--------|
| F1-Score (Macro) | ≥ 0.80 |
| ROC-AUC | ≥ 0.85 |
| Precision | ≥ 0.80 |
| Recall | ≥ 0.80 |

---

## 📝 Configuration

Edit `src/config.py` to customize:
- Random seed (reproducibility)
- Train/validation/test split ratios
- Audio preprocessing parameters (sample rate, window size)
- Model hyperparameters

---

## 🧪 Testing Reproducibility

To verify the entire pipeline is reproducible:

```bash
# 1. Delete all data & models
rm -rf data/* models/*

# 2. Download fresh data
python scripts/download_data.py

# 3. Retrain model
python scripts/train.py

# 4. Evaluate
python scripts/evaluate.py

# Metrics should match original run
```

---

## 🐛 Troubleshooting

### Issue: "AZURE_SAS_TOKEN not found"
**Solution**: Make sure `.env` file exists and contains your SAS token. Never hardcode it in scripts.

### Issue: "Audio file not found after download"
**Solution**: Verify Azure connection by running:
```bash
python -c "from src.data_loader import AzureBlobLoader; loader = AzureBlobLoader(); print(loader.list_files())"
```

### Issue: "Out of memory during training"
**Solution**: Reduce batch size or use gradient accumulation:
```bash
python scripts/train.py --batch_size 8 --accumulation_steps 4
```

---

## 📚 References

- **Audio Processing**: [librosa documentation](https://librosa.org/)
- **Vietnamese NLP**: [underthesea documentation](https://underthesea.readthedocs.io/)
- **Transformers**: [Hugging Face documentation](https://huggingface.co/docs/transformers/)
- **Azure Blob**: [azure-storage-blob documentation](https://learn.microsoft.com/en-us/python/api/azure-storage-blob/)

---

## 📧 Submission

When complete, upload deliverables to Azure Blob:
```bash
python scripts/upload_results.py
```

Then email supervisor with:
1. ✅ Upload confirmation
2. 📊 Best F1-score & ROC-AUC
3. 📝 One-paragraph methodology summary

---

## 📄 License

Internal project for ASR Team

---

**Last Updated**: June 2, 2026  
**Status**: In Development 🚀
