"""
Configuration & Constants

This module centralizes all project configuration parameters.
Load from environment variables and .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


class Config:
    """Central configuration class"""

    # Azure Blob Storage
    AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
    AZURE_CONTAINER_NAME = "asr-data"  # Will be confirmed when SAS token is provided
    AZURE_STORAGE_ACCOUNT = "your_storage_account"  # Will be provided

    # Data Configuration
    DATA_DIR = str(DATA_DIR)
    AUDIO_DIR = str(DATA_DIR / "audio")
    TRANSCRIPT_DIR = str(DATA_DIR / "transcripts")
    LABELS_FILE = str(DATA_DIR / "labels.csv")

    # Models
    MODELS_DIR = str(MODELS_DIR)
    REPORTS_DIR = str(REPORTS_DIR)

    # Training Configuration
    RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
    TEST_SIZE = 0.15
    VAL_SIZE = 0.15
    TRAIN_SIZE = 1.0 - TEST_SIZE - VAL_SIZE

    # Audio Processing
    SAMPLE_RATE = 16000
    DURATION = None  # Load entire audio file
    MONO = True

    # Feature Extraction
    N_MFCC = 13
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512

    # Model Training
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 10

    # Evaluation
    STRATIFIED_KFOLD = True
    N_SPLITS = 5

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = str(PROJECT_ROOT / "logs" / "asr_classifier.log")

    @classmethod
    def validate(cls):
        """Validate essential configuration"""
        if not cls.AZURE_SAS_TOKEN:
            print(
                "⚠️  WARNING: AZURE_SAS_TOKEN not set. "
                "Set it in .env file before downloading data."
            )
        return True

    @classmethod
    def to_dict(cls):
        """Return configuration as dictionary"""
        return {
            k: v for k, v in cls.__dict__.items() if not k.startswith("_")
        }


# Create logs directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Validate configuration on import
Config.validate()
