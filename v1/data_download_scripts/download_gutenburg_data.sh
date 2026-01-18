#!/bin/bash
# ==============================================================================
# Script Name: download_gutenburg_data.sh
# Description: Downloads text files from Project Gutenberg for use in LSTM
#              text generation training. The script is designed to be robust,
#              idempotent, and cross-platform compatible.
#
# Usage:       bash download_gutenburg_data.sh <BOOK_ID> <DATA_NAME> [OPTIONS]
#
# Arguments:
#   BOOK_ID    The Project Gutenberg book ID (e.g., 11339 for Aesop's Fables)
#   DATA_NAME  Name for the data directory (e.g., 'aesop')
#
# Options:
#   --force    Force re-download even if file already exists and is valid
#   --marker   Text marker to validate content (e.g., "THE FOX AND THE GRAPES")
#
# Examples:
#   # Download Aesop's Fables (Book ID 11339)
#   bash download_gutenburg_data.sh 11339 aesop
#
#   # Force re-download with content validation
#   bash download_gutenburg_data.sh 11339 aesop --force --marker "THE FOX AND THE GRAPES"
#
# Output:
#   Downloads text to 'v1/data/<DATA_NAME>/data.txt' relative to project root.
#
# Dependencies:
#   - bash
#   - One of: curl, wget, or python3 (auto-detected in that order)
#   - grep (for content validation)
#
# Exit Codes:
#   0 - Success (file downloaded or already exists)
#   1 - Error (missing arguments, download failed, or validation failed)
#
# Author: Generative Deep Learning Project
# License: GPL-3.0
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ==============================================================================
# 1. Configuration and Path Resolution
# ==============================================================================

# Determine the absolute path to the project root.
# Script is in 'v1/data_download_scripts/', so project root is '../..'
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="$PROJECT_ROOT/v1"

# ==============================================================================
# 2. Argument Parsing
# ==============================================================================

# Initialize variables
BOOK_ID=""
DATA_NAME=""
FORCE=false
MARKER=""

# Parse positional and optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        --marker)
            MARKER="$2"
            shift 2
            ;;
        *)
            # Positional arguments
            if [[ -z "$BOOK_ID" ]]; then
                BOOK_ID="$1"
            elif [[ -z "$DATA_NAME" ]]; then
                DATA_NAME="$1"
            else
                echo "Error: Unknown argument '$1'"
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$BOOK_ID" ]] || [[ -z "$DATA_NAME" ]]; then
    echo "Error: Missing required arguments."
    echo ""
    echo "Usage: bash download_gutenburg_data.sh <BOOK_ID> <DATA_NAME> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  BOOK_ID    Project Gutenberg book ID (e.g., 11339)"
    echo "  DATA_NAME  Name for data directory (e.g., 'aesop')"
    echo ""
    echo "Options:"
    echo "  --force    Force re-download even if file exists"
    echo "  --marker   Text to validate in downloaded content"
    echo ""
    echo "Examples:"
    echo "  bash download_gutenburg_data.sh 11339 aesop"
    echo "  bash download_gutenburg_data.sh 11339 aesop --marker 'THE FOX AND THE GRAPES'"
    exit 1
fi

# Build paths and URL
DATA_DIR="$BASE_DIR/data/$DATA_NAME"
TARGET_FILE="$DATA_DIR/data.txt"
URL="https://www.gutenberg.org/cache/epub/$BOOK_ID/pg$BOOK_ID.txt"

# ==============================================================================
# 3. Download Tool Detection
# ==============================================================================
# Checks for available download tools in order of preference:
# 1. curl (most common, good error handling)
# 2. wget (fallback, widely available on Linux)
# 3. python3 (universal fallback using urllib)

detect_download_tool() {
    if command -v curl &> /dev/null; then
        echo "curl"
    elif command -v wget &> /dev/null; then
        echo "wget"
    elif command -v python3 &> /dev/null; then
        echo "python3"
    else
        echo "none"
    fi
}

DL_TOOL=$(detect_download_tool)

if [[ "$DL_TOOL" == "none" ]]; then
    echo "Error: No download tool found."
    echo "Please install one of: curl, wget, or python3"
    exit 1
fi

echo "Using download tool: $DL_TOOL"

# ==============================================================================
# 4. Directory Setup
# ==============================================================================

if [[ ! -d "$DATA_DIR" ]]; then
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# ==============================================================================
# 5. Idempotency Check
# ==============================================================================
# Skip download if file exists, is valid, and --force is not set

validate_content() {
    local file="$1"
    local marker="$2"
    
    # Check file size (must be > 1KB)
    local file_size
    file_size=$(wc -c < "$file" 2>/dev/null || echo 0)
    
    if [[ "$file_size" -lt 1024 ]]; then
        echo "invalid_size"
        return
    fi
    
    # Check for marker if provided
    if [[ -n "$marker" ]]; then
        if grep -q "$marker" "$file" 2>/dev/null; then
            echo "valid"
        else
            echo "invalid_content"
        fi
    else
        echo "valid"
    fi
}

if [[ -f "$TARGET_FILE" ]] && [[ "$FORCE" == false ]]; then
    echo "File already exists: $TARGET_FILE"
    echo "Validating content..."
    
    VALIDATION=$(validate_content "$TARGET_FILE" "$MARKER")
    
    case $VALIDATION in
        valid)
            FILE_SIZE=$(wc -c < "$TARGET_FILE")
            echo "✓ File is valid (Size: $FILE_SIZE bytes)"
            echo "Skipping download. Use --force to re-download."
            exit 0
            ;;
        invalid_size)
            echo "⚠ File exists but is too small. Re-downloading..."
            ;;
        invalid_content)
            echo "⚠ File exists but content validation failed. Re-downloading..."
            ;;
    esac
elif [[ "$FORCE" == true ]]; then
    echo "Force mode: Re-downloading file..."
fi

# ==============================================================================
# 6. Download Execution
# ==============================================================================

echo ""
echo "Downloading from Project Gutenberg..."
echo "  Book ID: $BOOK_ID"
echo "  URL: $URL"
echo "  Destination: $TARGET_FILE"
echo ""

download_file() {
    local url="$1"
    local output="$2"
    local tool="$3"
    
    case $tool in
        curl)
            curl -fL --progress-bar -o "$output" "$url"
            ;;
        wget)
            wget --progress=bar:force -O "$output" "$url"
            ;;
        python3)
            echo "Using Python urllib for download..."
            python3 -c "
import urllib.request
import sys
url, output = sys.argv[1], sys.argv[2]
print(f'Downloading {url}...')
urllib.request.urlretrieve(url, output)
print('Download complete.')
" "$url" "$output"
            ;;
    esac
}

if download_file "$URL" "$TARGET_FILE" "$DL_TOOL"; then
    echo ""
    echo "Download completed successfully."
else
    echo ""
    echo "Error: Download failed!"
    echo "Please check:"
    echo "  - Your internet connection"
    echo "  - The Book ID ($BOOK_ID) is valid on gutenberg.org"
    exit 1
fi

# ==============================================================================
# 7. Post-Download Verification
# ==============================================================================

if [[ -f "$TARGET_FILE" ]]; then
    VALIDATION=$(validate_content "$TARGET_FILE" "$MARKER")
    
    if [[ "$VALIDATION" == "valid" ]]; then
        echo ""
        echo "========================================================"
        echo "✓ Success! Dataset downloaded and verified."
        echo "========================================================"
        echo "  File: $TARGET_FILE"
        ls -lh "$TARGET_FILE"
        echo "========================================================"
    else
        echo ""
        echo "Error: Downloaded file failed validation."
        echo "The file may be corrupted or the Book ID may be incorrect."
        exit 1
    fi
else
    echo "Error: File not found after download."
    exit 1
fi
