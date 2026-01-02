#!/bin/bash
# ==============================================================================
# Script Name: download_camel_data.sh
# Description: This script downloads the 'camel' dataset from the Google QuickDraw
#              repository for use in Generative Adversarial Network (GAN) training.
#              It is designed to be robust, idempotent, and easy to use.
#
# Usage:       ./download_camel_data.sh [OPTIONS]
#
# Options:
#   --force    Force re-download of the dataset even if it already exists.
#              Useful if the existing file is corrupted or you want to ensure
#              you have the latest version.
#
# Dependencies:
#   - bash
#   - curl OR wget (automatically detected)
#   - wc (for file size checking)
#
# Output:
#   Downloads 'camel.npy' to 'v1/data/camel/' directory relative to the project root.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status.
set -e

# ==============================================================================
# 1. Configuration and Path Resolution
# ==============================================================================

# Determine the absolute path to the project root.
# This ensures the script works correctly regardless of which directory it is called from.
# We assume the script is located in 'v1/scripts/', so we go up two levels ('../..').
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Define the base directory for version 1 of the project.
BASE_DIR="$PROJECT_ROOT/v1"

# Define the specific data directory for the camel dataset.
DATA_DIR="$BASE_DIR/data/camel"

# Define the full path for the target file.
TARGET_FILE="$DATA_DIR/camel.npy"

# URL for the Google QuickDraw 'camel' numpy bitmap dataset.
URL="https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/camel.npy"

# Initialize the FORCE flag to false by default.
FORCE=false

# ==============================================================================
# 2. Argument Parsing
# ==============================================================================

# Check for the --force flag in the first argument.
if [[ "$1" == "--force" ]]; then
    FORCE=true
    echo "Force mode enabled: Will overwrite existing files."
fi

# ==============================================================================
# 3. Dependency Checking
# ==============================================================================

# Check if 'curl' is available. If so, configure it to fail on error (-f), 
# follow redirects (-L), and write to output (-o).
if command -v curl &> /dev/null; then
    DL_CMD="curl -f -L -o"
# If 'curl' is not found, check for 'wget'. Configure it to write to output (-O).
elif command -v wget &> /dev/null; then
    DL_CMD="wget -O"
else
    # If neither tool is found, print an error and exit with status 1.
    echo "Error: Neither 'curl' nor 'wget' found. Please install one to continue."
    exit 1
fi

# ==============================================================================
# 4. Directory Setup
# ==============================================================================

# Check if the data directory exists. If not, create it.
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory at: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# ==============================================================================
# 5. Idempotency Check
# ==============================================================================

# Check if the target file already exists and FORCE is false.
if [ -f "$TARGET_FILE" ] && [ "$FORCE" = false ]; then
    echo "File $TARGET_FILE already exists."
    echo "Checking file validity..."
    
    # Get the file size in bytes.
    FILE_SIZE=$(wc -c < "$TARGET_FILE")
    
    # Basic validation: Check if file size is greater than 1KB.
    # A corrupted download or empty file would likely be smaller.
    if [ "$FILE_SIZE" -gt 1024 ]; then
        echo "File seems valid (Size: $FILE_SIZE bytes)."
        echo "Skipping download to save bandwidth."
        echo "Tip: Run with --force to re-download if needed."
        exit 0
    else
        echo "Warning: File exists but appears corrupted or too small ($FILE_SIZE bytes)."
        echo "Proceeding with re-download."
    fi
elif [ "$FORCE" = true ]; then
    # If FORCE is true, we simply proceed to the download section.
    echo "Re-downloading file as requested..."
fi

# ==============================================================================
# 6. Download Execution
# ==============================================================================

echo "Downloading camel.npy from Google Cloud Storage..."
echo "Source: $URL"
echo "Destination: $TARGET_FILE"

# Execute the download command constructed in Section 3.
if $DL_CMD "$TARGET_FILE" "$URL"; then
    echo "Download completed successfully."
else
    echo "Error: Download failed!"
    echo "Please check your internet connection and the URL."
    exit 1
fi

# ==============================================================================
# 7. Verification
# ==============================================================================

# Verify that the file exists after the download attempt.
if [ -f "$TARGET_FILE" ]; then
    echo "========================================================"
    echo "Success! Dataset setup complete."
    echo "File Location: $TARGET_FILE"
    # Show the file details (size, permissions, date) to the user.
    ls -lh "$TARGET_FILE"
    echo "========================================================"
else
    # If the file is missing despite a successful command exit code (unlikely but possible), fail.
    echo "Error: Critical failure. File not found after download."
    exit 1
fi
