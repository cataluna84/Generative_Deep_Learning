# AGENTS.md - V1 (1st Edition)

## Context

This folder contains notebooks from "Generative Deep Learning" 1st Edition experiments.

## Do

- Import models from `src.models.*`
- Import utils from `src.utils.*`
- Use root `wandb_utils.py` for W&B: `sys.path.insert(0, '..'); from wandb_utils import *`
- Enable GPU memory growth in first cell

## Don't

- Don't modify TF 1.x style code unless updating
- Don't commit model weights to git

## Structure

```
v1/
├── *.ipynb          # Flat notebook structure
├── *.py             # Scripts
└── src/
    ├── models/
    └── utils/
```
