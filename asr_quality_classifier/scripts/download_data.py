#!/usr/bin/env python
"""
Download Data from Azure Blob Storage

Usage:
    python scripts/download_data.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.data_loader import AzureBlobLoader, DataManager
from src.utils import setup_logger, print_config

logger = setup_logger(__name__, log_level=Config.LOG_LEVEL)


def main():
    """Main download function"""

    logger.info("🚀 Starting data download from Azure Blob Storage...")

    # Print configuration
    print_config(
        {"Azure Storage Account": Config.AZURE_STORAGE_ACCOUNT,
         "Container": Config.AZURE_CONTAINER_NAME,
         "Data Directory": Config.DATA_DIR},
        title="DOWNLOAD CONFIGURATION"
    )

    # Check if SAS token is set
    if not Config.AZURE_SAS_TOKEN:
        logger.error(
            "❌ AZURE_SAS_TOKEN not set!\n"
            "   Please add your SAS token to .env file:\n"
            "   AZURE_SAS_TOKEN=your_sas_token_here"
        )
        sys.exit(1)

    try:
        # Initialize Azure Blob loader
        logger.info("🔌 Connecting to Azure Blob Storage...")
        loader = AzureBlobLoader()

        # Download labels.csv
        logger.info("\n📥 Downloading labels.csv...")
        loader.download_file("data/labels.csv", Config.LABELS_FILE, overwrite=True)

        # Download audio files
        logger.info("\n📥 Downloading audio files from data/audio/...")
        audio_success, audio_fail = loader.download_folder(
            "data/audio/",
            Config.AUDIO_DIR,
            overwrite=False,
        )

        # Download transcript files
        logger.info("\n📥 Downloading transcripts from data/transcripts/...")
        trans_success, trans_fail = loader.download_folder(
            "data/transcripts/",
            Config.TRANSCRIPT_DIR,
            overwrite=False,
        )

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOAD SUMMARY".center(60))
        logger.info("=" * 60)
        logger.info(f"Audio files:     {audio_success} ✅, {audio_fail} ❌")
        logger.info(f"Transcripts:     {trans_success} ✅, {trans_fail} ❌")
        logger.info("=" * 60)

        # Print data statistics
        data_manager = DataManager()
        data_manager.print_data_summary()

        logger.info("✅ Data download complete!")

    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
