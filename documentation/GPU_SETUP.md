# GPU Setup Guide

This project is configured to use TensorFlow 2.20.0 with GPU support enabled via `tensorflow[and-cuda]`.

## Prerequisites

- Linux system with NVIDIA GPU
- CUDA drivers installed on the host machine

## Installation

We use `uv` for package management, which handles the complex dependency resolution for TensorFlow and CUDA.

To install the dependencies including CUDA libraries:

```bash
uv pip install -r requirements.txt
# OR if using uv sync
uv sync
```

## Verification

To verify that TensorFlow can see your GPU, run the provided verification script:

```bash
uv run python verify_gpu.py
```

This should output a list of available GPUs.
