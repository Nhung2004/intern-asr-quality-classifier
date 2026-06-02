"""
Utility Functions

Common utilities for logging, random seed setting, and helpers.
"""

import os
import sys
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import Config


def setup_logger(
    name: str = "asr_classifier",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logger with file and console handlers.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, uses Config.LOG_FILE

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    log_file = log_file or Config.LOG_FILE
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_level))
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_directory(path: str) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.

    Args:
        file_path: Path to file

    Returns:
        Formatted size string (e.g., "5.2 MB")
    """
    size_bytes = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def print_config(config_dict: dict, title: str = "Configuration") -> None:
    """
    Pretty print configuration dictionary.

    Args:
        config_dict: Configuration dictionary
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    for key, value in sorted(config_dict.items()):
        if isinstance(value, str) and len(value) > 50:
            print(f"{key:.<40} {value[:47]}...")
        else:
            print(f"{key:.<40} {value}")
    print(f"{'='*60}\n")
