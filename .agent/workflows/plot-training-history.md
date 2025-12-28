---
description: Plot training history after model training completes
---
# Plot Training History Workflow

This workflow should be added to every training notebook after the `model.fit()` or `AE.train()` call completes.

## Steps

1. Add this cell after training completes (before `wandb.finish()` if using W&B):

```python
# Plot training history with improved visualization
import matplotlib.pyplot as plt

history = AE.model.history.history  # Or model.history.history for standard Keras

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Loss (linear scale with log option for wide ranges)
axes[0].plot(history['loss'], 'b-', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss Over Epochs')
axes[0].grid(True, alpha=0.3)

# Plot 2: Learning Rate (LOG SCALE for visibility of reductions)
if 'learning_rate' in history:
    axes[1].semilogy(history['learning_rate'], 'r-', linewidth=2)  # Log scale!
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate (log scale)')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, which='both', alpha=0.3)
else:
    axes[1].text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
    axes[1].set_title('Learning Rate (Not Available)')

plt.tight_layout()
plt.show()

# Print summary
print(f"\n{'='*50}")
print("TRAINING SUMMARY")
print(f"{'='*50}")
print(f"  Initial Loss  : {history['loss'][0]:.6f}")
print(f"  Final Loss    : {history['loss'][-1]:.6f}")
print(f"  Min Loss      : {min(history['loss']):.6f} (Epoch {history['loss'].index(min(history['loss'])) + 1})")
print(f"  Total Epochs  : {len(history['loss'])}")
if 'learning_rate' in history:
    print(f"  Final LR      : {history['learning_rate'][-1]:.2e}")
print(f"{'='*50}")
```

2. Run the cell to visualize training progress.

## Key Features

- **Loss plot**: Linear scale showing convergence curve
- **LR plot**: Uses `semilogy()` for **log scale** - makes LR reductions visible even when small
- **Summary**: Displays initial, final, and minimum loss values with epoch number

## Interpreting Results

| Pattern | Meaning |
|---------|---------|
| Smooth decay | Good convergence |
| Flat loss curve | Model at capacity or LR too small |
| Rising loss | LR too high, instability |
| Stepped LR | ReduceLROnPlateau working |

## Notes

- Use `semilogy()` for LR to see reductions clearly (linear scale can hide small changes)
- If early stopping triggered, actual epochs < configured epochs
