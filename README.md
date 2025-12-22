# Generative Deep Learning

Inspired by the official code repository, for examples in the O'Reilly book 'Generative Deep Learning'

++ Experiments & research in progress

https://learning.oreilly.com/library/view/generative-deep-learning/9781492041931/

https://www.amazon.com/Generative-Deep-Learning-Teaching-Machines/dp/1492041947/ref=sr_1_1

## TensorFlow

This project uses TensorFlow 2.16+ with Keras 3.0+ and is compatible with Python 3.13+.

## Structure

This repository is structured as follows:

The notebooks for each chapter are in the root of the repository, prefixed with the chapter number.

The `data` folder is where to download relevant data sources
The `run` folder stores output from the generative models
The `utils` folder stores useful functions that are sourced by the main notebooks

## Getting started

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable package management.

### Install uv

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Set up the project

```bash
# Create virtual environment and install dependencies
uv venv
uv pip install -r requirements.txt

# Activate the environment
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# Start JupyterLab
jupyter lab
```
