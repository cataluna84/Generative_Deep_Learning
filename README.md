# Generative Deep Learning

Experiments based on O'Reilly's "Generative Deep Learning" books.

## Structure

```
├── v1/              # 1st Edition (2019) - 22 notebooks
├── v2/              # 2nd Edition (2023) - Organized by chapter
├── wandb_utils.py   # Shared W&B integration
├── documentation/   # Setup guides
└── data/            # Downloaded datasets
```

## Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup environment
uv sync

# Run Jupyter
uv run jupyter lab
```

## TensorFlow

- Python 3.13+
- TensorFlow 2.20 with CUDA
- See [GPU Setup](documentation/GPU_SETUP.md)

## Versions

| Version | Book Edition | Notebooks |
|---------|-------------|-----------|
| v1/ | 1st Edition (2019) | 22 |
| v2/ | 2nd Edition (2023) | 40+ |
