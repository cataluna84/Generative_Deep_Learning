#!/bin/bash
# Script to download CelebA dataset from Kaggle and setup for VAE training
# Usage: ./download_celeba_kaggle.sh

# Exit on error
set -e

# 1. Locate and load .env from project root
# Assuming script is in v1/scripts/, root is ../../
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading credentials from $ENV_FILE..."
    export $(grep -v '^#' "$ENV_FILE" | xargs)
else
    echo "Error: .env file not found at $ENV_FILE"
    echo "Please ensure .env exists with KAGGLE_USERNAME and KAGGLE_KEY"
    exit 1
fi

# Check credentials
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "Error: KAGGLE_USERNAME or KAGGLE_KEY not found in .env"
    exit 1
fi

# 2. Check for kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "Kaggle CLI not found. Installing..."
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

# 3. Define paths
# Target: v1/data/img_align_celeba/images/*.jpg
# We want to run from v1/ directory ideally, or relative to script
BASE_DIR="$PROJECT_ROOT/v1"
DATA_DIR="$BASE_DIR/data"
TEMP_DIR="$DATA_DIR/temp_celeba_download"
TARGET_IMG_DIR="$DATA_DIR/img_align_celeba/images"

# Start clean
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"
mkdir -p "$TARGET_IMG_DIR"

# Determine execution command
if command -v uv &> /dev/null; then
    KAGGLE_CMD="uv run kaggle"
else
    KAGGLE_CMD="kaggle"
fi

echo "Downloading CelebA dataset from Kaggle (forcing clean download)..."
# Download specifically the img_align_celeba.zip file if possible, or the whole dataset
# The dataset ID is jessicali9530/celeba-dataset
$KAGGLE_CMD datasets download -d jessicali9530/celeba-dataset -p "$TEMP_DIR" --force

echo "Verifying zip integrity..."
DOWNLOADED_ZIP=$(find "$TEMP_DIR" -name "*.zip" | head -n 1)

if [ -z "$DOWNLOADED_ZIP" ]; then
    echo "Error: No zip file downloaded."
    exit 1
fi

# verification
if ! python3 -m zipfile -t "$DOWNLOADED_ZIP"; then
    echo "Error: Downloaded zip file is corrupted."
    exit 1
fi

echo "Unzipping..."
# The downloaded file is likely celeba-dataset.zip or img_align_celeba.zip depending on how kaggle packages it

# Let's check what we got
# DOWNLOADED_ZIP is already set and checked above

# Unzip specifically the img_align_celeba/ folder contents
echo "Extracting $DOWNLOADED_ZIP..."

if command -v unzip &> /dev/null; then
    unzip -q "$DOWNLOADED_ZIP" "img_align_celeba/img_align_celeba/*" -d "$TEMP_DIR"
else
    echo "unzip command not found, using python..."
    # Python unzip fallback (extracts everything because partial extract with glob is harder in one-liner, but we can filter later or just extract all)
    # Extracting all for simplicity in fallback
    python3 -c "import zipfile, sys; 
with zipfile.ZipFile(sys.argv[1], 'r') as z: 
    z.extractall(sys.argv[2])" "$DOWNLOADED_ZIP" "$TEMP_DIR"
fi

# Move files to target
echo "Arranging files in $TARGET_IMG_DIR..."
# The unzip likely created $TEMP_DIR/img_align_celeba/img_align_celeba/
SOURCE_IMGS="$TEMP_DIR/img_align_celeba/img_align_celeba"

if [ -d "$SOURCE_IMGS" ]; then
    echo "Moving source directory to target..."
    # Remove the empty target dir created earlier (or existing dir) and replace with source
    # This avoids "Argument list too long" error with mv *
    rm -rf "$TARGET_IMG_DIR"
    mv "$SOURCE_IMGS" "$TARGET_IMG_DIR"
else
    # Fallback: maybe structure is different, try finding jpgs
    echo "Standard path not found, searching for jpgs..."
    # Use find with xargs for efficiency and to avoid ARG_MAX
    # Using mv -t (target directory) which is standard on Linux 
    find "$TEMP_DIR" -name "*.jpg" -print0 | xargs -0 mv -t "$TARGET_IMG_DIR/"
fi

# 4. Cleanup
echo "Cleaning up temp files..."
rm -rf "$TEMP_DIR"

echo "Success! CelebA images are ready at: $TARGET_IMG_DIR"
ls -1 "$TARGET_IMG_DIR" | head -n 5
echo "..."
echo "Total images: $(ls -1 "$TARGET_IMG_DIR" | wc -l)"
