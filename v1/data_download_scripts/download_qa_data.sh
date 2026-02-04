#!/bin/bash
set -e

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="$PROJECT_ROOT/v1"
QA_ARCHIVE="$BASE_DIR/data/qa.tar.xz"
GLOVE_DIR="$BASE_DIR/data/glove"
GLOVE_URL="https://nlp.stanford.edu/data/glove.6B.zip"

# Parse args
FORCE=false
[[ "$1" == "--force" ]] && FORCE=true

echo ""
echo "╔═══════════════════════════════════════════════════════╗"
echo "║  QA Data Setup for Question-Answering Training       ║"
echo "╚═══════════════════════════════════════════════════════╝"
echo ""

# Step 1: Extract QA data
echo "=========================================================="
echo "STEP 1: Extracting QA Dataset"
echo "=========================================================="

if [[ -f "$BASE_DIR/data/qa/train.csv" ]] && [[ -f "$BASE_DIR/data/qa_test/my_test.csv" ]] && [[ "$FORCE" == false ]]; then
    echo "✓ QA data already extracted"
    ls -lh "$BASE_DIR/data/qa/train.csv"
    ls -lh "$BASE_DIR/data/qa_test/my_test.csv"
else
    echo "Extracting $QA_ARCHIVE..."
    tar -xJf "$QA_ARCHIVE" -C "$BASE_DIR/data"
    
    # Create qa_test directory and copy test file (write.py expects this structure)
    echo "Setting up test data structure..."
    mkdir -p "$BASE_DIR/data/qa_test"
    cp "$BASE_DIR/data/qa/test.csv" "$BASE_DIR/data/qa_test/my_test.csv"
    
    echo "✓ Extraction complete"
    ls -lh "$BASE_DIR/data/qa/train.csv"
    ls -lh "$BASE_DIR/data/qa_test/my_test.csv"
fi
echo ""

# Step 2: Download GloVe
echo "=========================================================="
echo "STEP 2: Downloading GloVe Embeddings"
echo "=========================================================="

mkdir -p "$GLOVE_DIR"

if [[ -f "$GLOVE_DIR/glove.6B.100d.txt" ]] && [[ "$FORCE" == false ]]; then
    echo "✓ GloVe already downloaded"
    ls -lh "$GLOVE_DIR/glove.6B.100d.txt"
else
    echo "Downloading GloVe 6B (~822MB)..."
    echo "This may take several minutes..."
    if command -v curl &> /dev/null; then
        curl -fL --progress-bar -o "$GLOVE_DIR/glove.6B.zip" "$GLOVE_URL"
    elif command -v wget &> /dev/null; then
        wget --progress=bar:force -O "$GLOVE_DIR/glove.6B.zip" "$GLOVE_URL"
    else
        echo "Error: curl or wget required"
        exit 1
    fi
    
    echo "Extracting GloVe..."
    unzip -q "$GLOVE_DIR/glove.6B.zip" -d "$GLOVE_DIR" glove.6B.100d.txt
    rm "$GLOVE_DIR/glove.6B.zip"
    echo "✓ Download complete"
    ls -lh "$GLOVE_DIR/glove.6B.100d.txt"
fi
echo ""

# Step 3: Create trimmed embeddings
echo "=========================================================="
echo "STEP 3: Creating Trimmed Embeddings"
echo "=========================================================="

if [[ -f "$GLOVE_DIR/glove.6B.100d.trimmed.txt" ]] && [[ "$FORCE" == false ]]; then
    echo "✓ Trimmed embeddings already exist"
    ls -lh "$GLOVE_DIR/glove.6B.100d.trimmed.txt"
else
    echo "Running trim_embeddings()..."
    cd "$BASE_DIR/notebooks"
    python3 << 'PYTHON_EOF'
import sys
sys.path.insert(0, '..')
from src.utils.write import trim_embeddings
print('Trimming embeddings...')
trim_embeddings()
print('✓ Trimming complete')
PYTHON_EOF
    
    echo "✓ Trimmed embeddings created"
    ls -lh "$GLOVE_DIR/glove.6B.100d.trimmed.txt"
    VOCAB_SIZE=$(wc -l < "$GLOVE_DIR/glove.6B.100d.trimmed.txt")
    echo "  Vocabulary size: $VOCAB_SIZE words"
fi
echo ""

echo "=========================================================="
echo "✓ Setup Complete!"
echo "=========================================================="
echo ""
echo "You can now run: v1/notebooks/06_02_qa_train.ipynb"
echo ""
