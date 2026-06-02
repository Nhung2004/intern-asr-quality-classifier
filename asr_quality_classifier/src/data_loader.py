"""
Data Loader - Azure Blob Storage Integration

Handles downloading data from Azure Blob Storage and managing local datasets.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from tqdm import tqdm

from src.config import Config
from src.utils import setup_logger, ensure_directory

logger = setup_logger(__name__)


class AzureBlobLoader:
    """
    Manages connection and data download from Azure Blob Storage.
    """

    def __init__(
        self,
        sas_token: Optional[str] = None,
        container_name: Optional[str] = None,
    ):
        """
        Initialize Azure Blob Storage client.

        Args:
            sas_token: SAS token for authentication (reads from Config if not provided)
            container_name: Blob container name
        """
        self.sas_token = sas_token or Config.AZURE_SAS_TOKEN
        self.container_name = container_name or Config.AZURE_CONTAINER_NAME

        if not self.sas_token:
            raise ValueError(
                "AZURE_SAS_TOKEN not set. "
                "Please set it in .env file or pass as parameter."
            )

        # Initialize Blob Service Client
        # Note: Exact account URL will be provided with SAS token
        self.account_url = (
            f"https://{Config.AZURE_STORAGE_ACCOUNT}.blob.core.windows.net"
        )
        self.client = BlobServiceClient(
            account_url=self.account_url,
            credential=self.sas_token,
        )
        self.container_client = self.client.get_container_client(
            self.container_name
        )

        logger.info(
            f"✅ Connected to Azure Blob Storage: {self.account_url}/{self.container_name}"
        )

    def list_files(self, prefix: str = "") -> List[str]:
        """
        List all files in container with optional prefix filter.

        Args:
            prefix: Blob name prefix to filter (e.g., "data/audio/")

        Returns:
            List of blob names
        """
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            file_list = [blob.name for blob in blobs]
            logger.info(f"Found {len(file_list)} files with prefix '{prefix}'")
            return file_list
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            raise

    def download_file(
        self,
        blob_name: str,
        local_path: str,
        overwrite: bool = False,
    ) -> bool:
        """
        Download single file from Blob Storage to local path.

        Args:
            blob_name: Name of blob to download
            local_path: Local path to save file
            overwrite: Whether to overwrite existing file

        Returns:
            True if successful, False otherwise
        """
        try:
            local_path = Path(local_path)

            # Skip if file exists and overwrite=False
            if local_path.exists() and not overwrite:
                logger.debug(f"⏭️  File exists (skipping): {local_path}")
                return True

            # Create parent directory
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download blob
            blob_client = self.container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                download_stream = blob_client.download_blob()
                f.write(download_stream.readall())

            logger.debug(f"✅ Downloaded: {blob_name} → {local_path}")
            return True

        except Exception as e:
            logger.error(
                f"❌ Error downloading {blob_name}: {e}"
            )
            return False

    def download_folder(
        self,
        blob_prefix: str,
        local_dir: str,
        overwrite: bool = False,
    ) -> Tuple[int, int]:
        """
        Download all files with given prefix from Blob Storage.

        Args:
            blob_prefix: Prefix in Blob Storage (e.g., "data/audio/")
            local_dir: Local directory to save files
            overwrite: Whether to overwrite existing files

        Returns:
            Tuple of (successful_downloads, failed_downloads)
        """
        ensure_directory(local_dir)

        # List all files with prefix
        blob_files = self.list_files(prefix=blob_prefix)

        if not blob_files:
            logger.warning(f"No files found with prefix: {blob_prefix}")
            return 0, 0

        logger.info(f"Downloading {len(blob_files)} files...")

        success_count = 0
        fail_count = 0

        # Download with progress bar
        for blob_name in tqdm(blob_files, desc=f"Downloading from {blob_prefix}"):
            # Preserve folder structure
            relative_path = blob_name.replace(blob_prefix, "", 1)
            local_path = Path(local_dir) / relative_path

            if self.download_file(blob_name, str(local_path), overwrite):
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            f"✅ Download complete: {success_count} successful, {fail_count} failed"
        )
        return success_count, fail_count


class DataManager:
    """
    Manages local data loading and preprocessing.
    """

    def __init__(self, data_dir: str = Config.DATA_DIR):
        """
        Initialize data manager.

        Args:
            data_dir: Root data directory
        """
        self.data_dir = Path(data_dir)
        self.audio_dir = self.data_dir / "audio"
        self.transcript_dir = self.data_dir / "transcripts"
        self.labels_file = self.data_dir / "labels.csv"

        logger.info(f"📁 Data directory: {self.data_dir}")

    def load_labels(self) -> pd.DataFrame:
        """
        Load labels CSV file.

        Returns:
            DataFrame with columns: file_id, label
        """
        try:
            df = pd.read_csv(self.labels_file)
            logger.info(f"✅ Loaded {len(df)} labels from {self.labels_file}")

            # Validate binary labels (0, 1)
            unique_labels = df["label"].unique()
            if not set(unique_labels).issubset({0, 1}):
                logger.warning(
                    f"⚠️  Non-binary labels found: {unique_labels}. "
                    "Expected 0 (not usable) and 1 (usable)"
                )

            return df
        except FileNotFoundError:
            logger.error(f"❌ Labels file not found: {self.labels_file}")
            raise
        except Exception as e:
            logger.error(f"❌ Error loading labels: {e}")
            raise

    def get_audio_path(self, file_id: str) -> Path:
        """
        Get full path to audio file.

        Args:
            file_id: File identifier (without extension)

        Returns:
            Path to audio file
        """
        # Try .wav first, then .mp3
        for ext in [".wav", ".mp3"]:
            path = self.audio_dir / f"{file_id}{ext}"
            if path.exists():
                return path
        raise FileNotFoundError(f"Audio file not found for {file_id}")

    def get_transcript_path(self, file_id: str) -> Path:
        """
        Get full path to transcript file.

        Args:
            file_id: File identifier (without extension)

        Returns:
            Path to transcript file
        """
        path = self.transcript_dir / f"{file_id}.txt"
        if path.exists():
            return path
        raise FileNotFoundError(f"Transcript file not found for {file_id}")

    def get_data_statistics(self) -> dict:
        """
        Get statistics about downloaded data.

        Returns:
            Dictionary with data statistics
        """
        try:
            labels_df = self.load_labels()

            stats = {
                "total_samples": len(labels_df),
                "usable_count": (labels_df["label"] == 1).sum(),
                "not_usable_count": (labels_df["label"] == 0).sum(),
                "audio_files_found": len(list(self.audio_dir.glob("*"))),
                "transcript_files_found": len(list(self.transcript_dir.glob("*.txt"))),
            }

            # Calculate percentages
            total = stats["total_samples"]
            if total > 0:
                stats["usable_pct"] = (
                    stats["usable_count"] / total * 100
                )
                stats["not_usable_pct"] = (
                    stats["not_usable_count"] / total * 100
                )

            return stats
        except Exception as e:
            logger.error(f"Error getting data statistics: {e}")
            return {}

    def print_data_summary(self) -> None:
        """Print summary of downloaded data."""
        try:
            stats = self.get_data_statistics()
            print("\n" + "=" * 60)
            print("📊 DATA SUMMARY".center(60))
            print("=" * 60)
            print(f"Total Samples:        {stats.get('total_samples', 'N/A')}")
            print(f"  ✅ Usable:          {stats.get('usable_count', 'N/A')} "
                  f"({stats.get('usable_pct', 0):.1f}%)")
            print(f"  ❌ Not Usable:      {stats.get('not_usable_count', 'N/A')} "
                  f"({stats.get('not_usable_pct', 0):.1f}%)")
            print(f"Audio Files:          {stats.get('audio_files_found', 'N/A')}")
            print(f"Transcript Files:     {stats.get('transcript_files_found', 'N/A')}")
            print("=" * 60 + "\n")
        except Exception as e:
            logger.error(f"Error printing summary: {e}")
