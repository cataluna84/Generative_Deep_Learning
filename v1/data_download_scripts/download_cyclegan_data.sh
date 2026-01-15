#!/bin/bash

# Usage: bash download_cyclegan_data.sh <dataset_name>
# Example: bash download_cyclegan_data.sh apple2orange
# Available datasets: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data directory is always v1/data (one level up from data_download_scripts)
DATA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/data"

FILE=$1

if [[ -z "$FILE" ]]; then
    echo "Usage: bash download_cyclegan_data.sh <dataset_name>"
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

if [[ $FILE != "ae_photos" && $FILE != "apple2orange" && $FILE != "summer2winter_yosemite" &&  $FILE != "horse2zebra" && $FILE != "monet2photo" && $FILE != "cezanne2photo" && $FILE != "ukiyoe2photo" && $FILE != "vangogh2photo" && $FILE != "maps" && $FILE != "cityscapes" && $FILE != "facades" && $FILE != "iphone2dslr_flower" && $FILE != "ae_photos" ]]; then
    echo "Error: Invalid dataset name: $FILE"
    echo "Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos"
    exit 1
fi

# Create data directory if it doesn't exist
mkdir -p "$DATA_DIR"

# Official Berkeley Efros URL (direct source, no Kaggle keys needed)
URL=http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/$FILE.zip
ZIP_FILE="$DATA_DIR/$FILE.zip"
TARGET_DIR="$DATA_DIR/$FILE/"

echo "Downloading $FILE dataset to $DATA_DIR..."
# Download with progress (resume if possible)
if wget -c "$URL" -O "$ZIP_FILE"; then
    echo "Download completed."
else
    echo "Error: Download failed."
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "Verifying zip file integrity..."
# Verify zip file before unzipping
if command -v unzip &> /dev/null; then
    if ! unzip -tq "$ZIP_FILE"; then
        echo "Error: Zip file corrupted. Deleting..."
        rm "$ZIP_FILE"
        exit 1
    fi
else
    # Python fallback for verification
    if ! python3 -c "import zipfile, sys; sys.exit(1 if zipfile.ZipFile(sys.argv[1]).testzip() is not None else 0)" "$ZIP_FILE"; then
         echo "Error: Zip file corrupted (verified via Python). Deleting..."
         rm "$ZIP_FILE"
         exit 1
    fi
fi
echo "Verification successful."

echo "Unzipping..."
# Try unzip first, fall back to python if not available
if command -v unzip &> /dev/null; then
    unzip -q "$ZIP_FILE" -d "$DATA_DIR/"
else
    echo "'unzip' command not found, using Python fallback..."
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$ZIP_FILE" "$DATA_DIR/"
fi

rm "$ZIP_FILE"
echo "Done! Dataset available at $TARGET_DIR"
