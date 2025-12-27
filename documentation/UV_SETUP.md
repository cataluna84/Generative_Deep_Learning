# UV Setup Guide

This project uses [UV](https://docs.astral.sh/uv/) for Python package management.

## Prerequisites

- Ubuntu 24.04 LTS (WSL2 supported)
- NVIDIA GPU with drivers installed

## Quick Start

### 1. Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Python 3.13

```bash
uv python install 3.13
```

### 3. Sync Dependencies

```bash
cd /path/to/Generative_Deep_Learning_2nd_Edition
uv sync
```

This installs TensorFlow 2.20 with CUDA support and all required dependencies.

### 4. Run Jupyter Lab

```bash
uv run jupyter lab
```

## Common Commands

| Command | Description |
|---------|-------------|
| `uv sync` | Install/update all dependencies |
| `uv add <package>` | Add a new dependency |
| `uv run <command>` | Run command in virtual environment |
| `uv pip list` | List installed packages |

## Project Structure

```
├── pyproject.toml    # Dependencies and project config
├── uv.lock          # Locked dependencies (auto-generated)
├── .venv/           # Virtual environment (auto-created)
└── .python-version  # Python version pin
```

## Troubleshooting

### GPU Not Detected

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

If empty, check NVIDIA drivers:
```bash
nvidia-smi
```

### Memory Issues

For large models on GPUs with limited VRAM (e.g., RTX 2070 8GB), add to notebooks:

```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```
