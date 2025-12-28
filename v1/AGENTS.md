# AGENTS.md - V1 (1st Edition)

## Context

This directory contains the notebooks from the **1st Edition (2019)** of "Generative Deep Learning".

## Standard Workflow (Recursive Plan)

When working on notebooks in this directory, follow the **[Notebook Standardization Guide](../../documentation/NOTEBOOK_STANDARDIZATION.md)**.

**Summary of changes to apply recursively:**
1.  **Global Config**: Move `BATCH_SIZE`, `EPOCHS`, etc. to top-level variables.
2.  **W&B**: Initialize with global config and `learning_rate="auto"`.
3.  **LRFinder**: Insert the LRFinder workflow (Clone -> Find -> `plot_loss()` -> `get_optimal_lr()`) before main training.
4.  **Integration**: Update `wandb.config` with the optimal LR and use it in the main optimizer.
5.  **Finish**: Always call `wandb.finish()` at the end of the notebook.

## Do

- Import models from `src.models.*`
- Import utils from `src.utils.*`
- Use shared root utilities:
  - `utils.wandb_utils`: For Weights & Biases tracking.
  - `utils.callbacks`: For `LRFinder` and `get_lr_scheduler`.
- Enable GPU memory growth in the first cell (`tf.config.experimental.set_memory_growth`).

## Don't

- **Don't** hardcode training parameters (batch size, epochs) in `model.fit()`.
- **Don't** modify TF 1.x style code unless explicitly updating/refactoring.
- **Don't** commit model weights to git.

## Key Differences from V2

- Uses `src.models.*` imports (vs V2 which might use local or different structure).
- Often assumes being run from `v1/notebooks/` or similar, requiring `sys.path` adjustments.

## Structure

```
v1/
├── notebooks/       # The actual .ipynb files
├── scripts/         # Data download scripts
├── src/
│   ├── models/      # Legacy model definitions
│   └── utils/       # Legacy loaders
└── AGENTS.md        # This file
```
