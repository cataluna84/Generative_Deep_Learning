# Repository Reorganization Guide

## Structure Overview

```
Generative_Deep_Learning/
├── wandb_utils.py        # Shared W&B (root)
├── v1/                   # 1st Edition notebooks
│   ├── *.ipynb           # 22 notebooks
│   ├── *.py              # Companion scripts
│   └── src/
│       ├── models/       # AE, VAE, GAN, etc.
│       └── utils/        # Loaders, callbacks
├── v2/                   # 2nd Edition notebooks
│   ├── 02_deeplearning/
│   ├── 03_vae/
│   └── src/
└── documentation/
```

## Import Patterns

**From v1/ notebooks:**
```python
from src.models.VAE import VariationalAutoencoder
from src.utils.loaders import load_data
import sys; sys.path.insert(0, '..')
from wandb_utils import init_wandb
```

**From v2/ notebooks:**
```python
import sys; sys.path.insert(0, '../..')
from wandb_utils import init_wandb
```
