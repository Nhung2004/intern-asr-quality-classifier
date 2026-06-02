#!/usr/bin/env python
"""
Verify Project Structure

Check that all required files and folders are in place.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent


def verify_structure():
    """Verify project structure"""
    print("\n" + "=" * 70)
    print("PROJECT STRUCTURE VERIFICATION".center(70))
    print("=" * 70 + "\n")

    required_dirs = [
        "src",
        "scripts",
        "notebooks",
        "data",
        "models",
        "reports",
        "logs",
    ]

    required_files = {
        "Root": [
            ".env.example",
            ".gitignore",
            "README.md",
            "requirements.txt",
        ],
        "src": [
            "__init__.py",
            "config.py",
            "utils.py",
            "data_loader.py",
        ],
        "scripts": [
            "download_data.py",
            "extract_features.py",
            "train.py",
            "evaluate.py",
            "inference.py",
        ],
        "notebooks": [
            "eda.ipynb",
        ],
    }

    all_ok = True

    # Check directories
    print("📁 Checking directories...")
    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"  ✅ {dir_name}/")
        else:
            print(f"  ❌ {dir_name}/ (MISSING)")
            all_ok = False

    print()

    # Check files
    print("📄 Checking files...")
    for location, files in required_files.items():
        if location == "Root":
            base_path = PROJECT_ROOT
            label = ""
        else:
            base_path = PROJECT_ROOT / location
            label = f"{location}/"

        for file_name in files:
            file_path = base_path / file_name
            if file_path.exists() and file_path.is_file():
                print(f"  ✅ {label}{file_name}")
            else:
                print(f"  ❌ {label}{file_name} (MISSING)")
                all_ok = False

    print()

    # Summary
    print("=" * 70)
    if all_ok:
        print("✅ PROJECT STRUCTURE IS COMPLETE!".center(70))
        print("\nNext steps:".center(70))
        print("1. Set AZURE_SAS_TOKEN in .env file".center(70))
        print("2. Run: python scripts/download_data.py".center(70))
        print("3. Run: jupyter notebook notebooks/eda.ipynb".center(70))
    else:
        print("❌ SOME FILES ARE MISSING!".center(70))
    print("=" * 70 + "\n")

    return all_ok


if __name__ == "__main__":
    success = verify_structure()
    sys.exit(0 if success else 1)
