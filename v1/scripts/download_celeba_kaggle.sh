#!/bin/bash
# ==============================================================================
# Script Name: download_celeba_kaggle.sh
# Description: This script downloads the CelebA dataset from Kaggle, extracts it,
#              and organizes it into the directory structure required for VAE training.
#              It handles authentication via .env, dependency checks, and efficient
#              file operations for large datasets.
#
# Usage:       ./download_celeba_kaggle.sh
#
# Prerequisites:
#   - A Kaggle account and API key.
#   - A .env file in the project root containing KAGGLE_USERNAME and KAGGLE_KEY.
#
# Output:
#   - Images: v1/data/img_align_celeba/images/*.jpg
#   - Attributes: v1/data/img_align_celeba/list_attr_celeba.csv
# ==============================================================================

# Exit immediately if a command exits with a non-zero status to prevent cascading errors.
set -e

# ==============================================================================
# 1. Environment and Credentials Setup
# ==============================================================================

# Determine the absolute path to the project root.
# Assuming this script is located in 'v1/scripts/', we navigate up two levels ('../..').
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

# Load environment variables from the .env file if it exists.
if [ -f "$ENV_FILE" ]; then
    echo "Loading credentials from $ENV_FILE..."
    # Export variables from .env, ignoring key-value pairs starting with #
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please ensure .env exists with KAGGLE_USERNAME and KAGGLE_KEY defined."
    exit 1
fi

# Verify that Kaggle credentials are set.
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: KAGGLE_USERNAME or KAGGLE_KEY not found in .env"
    exit 1
fi

# ==============================================================================
# 2. Dependency Checking
# ==============================================================================

# Check if the 'kaggle' CLI tool is installed. If not, attempt to install it.
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
    # Prefer 'uv' for installation if available, otherwise fallback to 'python3'.
    if command -v uv &> /dev/null; then
         echo "Using uv to install kaggle..."
         uv pip install kaggle
    elif command -v python3 &> /dev/null; then
         echo "Using python3 to install kaggle..."
         python3 -m pip install kaggle
    else
         echo "Error: Neither uv nor python3 found. Cannot install kaggle."
         exit 1
    fi
fi

# Determine the command prefix for running kaggle.
if command -v uv &> /dev/null; then
    KAGGLE_CMD="uv run kaggle"
else
    KAGGLE_CMD="kaggle"
fi

# ==============================================================================
# 3. Path Defintions and Directory Setup
# ==============================================================================

# Define directory constants relative to the project structure.
BASE_DIR="$PROJECT_ROOT/v1"
DATA_DIR="$BASE_DIR/data"
TEMP_DIR="$DATA_DIR/temp_celeba_download"
TARGET_IMG_DIR="$DATA_DIR/img_align_celeba/images"

# Start with a clean state: remove temp dir if it exists and ensure target dirs exist.
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$TARGET_IMG_DIR"

# ==============================================================================
# 4. Dataset Download
# ==============================================================================

echo "Downloading CelebA dataset from Kaggle..."
# Download the dataset 'jessicali9530/celeba-dataset'. 
# We force a download to ensure we get a fresh copy to the temp directory.
$KAGGLE_CMD datasets download -d jessicali9530/celeba-dataset -p "$TEMP_DIR" --force

# ==============================================================================
# 5. Verification
# ==============================================================================

echo "Verifying zip integrity..."
# Find the downloaded zip file (name might vary).
DOWNLOADED_ZIP=$(find "$TEMP_DIR" -name "*.zip" | head -n 1)

if [ -z "$DOWNLOADED_ZIP" ]; then
    echo "Error: No zip file downloaded."
    exit 1
fi

# Verify zip file structure using python's zipfile module.
if ! python3 -m zipfile -t "$DOWNLOADED_ZIP"; then
    echo "Error: Downloaded zip file is corrupted."
    exit 1
fi

# ==============================================================================
# 6. Extraction and Organization
# ==============================================================================

echo "Unzipping and extracting relevant files..."

# We use Python for extraction to allow precise filtering.
# We only want to extract the 'img_align_celeba' folder and the attributes file.
# Extracting everything would waste space and time.
python3 -c "import zipfile, sys, os
zip_path = sys.argv[1]
extract_to = sys.argv[2]
print(f'Extracting contents from {zip_path} to {extract_to}...')
try:
    with zipfile.ZipFile(zip_path, 'r') as z:
        for file in z.namelist():
            # Filter for specific file patterns
            if file.startswith('img_align_celeba/img_align_celeba/') or 'list_attr_celeba' in file:
                # print(f'Extracting {file}...') # Commented out to reduce noise
                z.extract(file, extract_to)
except Exception as e:
    print(f'Extraction failed: {e}')
    sys.exit(1)
" "$DOWNLOADED_ZIP" "$TEMP_DIR"

# ==============================================================================
# 7. File Relocation
# ==============================================================================

echo "Arranging files in target directory: $TARGET_IMG_DIR"

# The extraction likely created a nested structure like '$TEMP_DIR/img_align_celeba/img_align_celeba/'
SOURCE_IMGS="$TEMP_DIR/img_align_celeba/img_align_celeba"

if [ -d "$SOURCE_IMGS" ]; then
    echo "Moving extracted source directory to target location..."
    # Efficiently move the directory.
    # We remove the target directory first to avoid issues with 'mv' merging directories.
    rm -rf "$TARGET_IMG_DIR"
    mkdir -p "$(dirname "$TARGET_IMG_DIR")"
    mv "$SOURCE_IMGS" "$TARGET_IMG_DIR"
else
    # Fallback Mechanism: If the directory structure isn't as expected, find and move all JPGs.
    echo "Standard extraction path not found. Searching for .jpg files..."
    # Use 'find' with 'xargs' to handle potential argument list length limits.
    find "$TEMP_DIR" -name "*.jpg" -print0 | xargs -0 mv -t "$TARGET_IMG_DIR/"
fi

echo "Organizing attributes file..."
# Find the attributes file (CSV or TXT) and move it to the parent data directory.
ATTR_FILE=$(find "$TEMP_DIR" -name "list_attr_celeba.csv" -o -name "list_attr_celeba.txt" | head -n 1)

if [ -n "$ATTR_FILE" ]; then
    echo "Found attributes file: $ATTR_FILE"
    mv "$ATTR_FILE" "$DATA_DIR/img_align_celeba/"
else
    echo "Warning: Attributes file (list_attr_celeba) not found!"
fi

# ==============================================================================
# 8. Cleanup and Final Status
# ==============================================================================

echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "========================================================"
echo "Success! CelebA dataset setup complete."
echo "Images Location: $TARGET_IMG_DIR"
echo "First 5 files:"
ls -1 "$TARGET_IMG_DIR" | head -n 5
echo "..."
echo "Total image count: $(ls -1 "$TARGET_IMG_DIR" | wc -l)"
echo "========================================================"
