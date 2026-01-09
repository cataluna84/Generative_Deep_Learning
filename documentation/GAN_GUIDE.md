# GAN Training Guide

Comprehensive guide for GAN training: metrics, stability analysis, triage, and W&B integration.

---

## W&B Initialization for GANs

GANs use custom training loops with per-epoch metrics logging.

```python
from utils.wandb_utils import init_wandb, define_wgan_charts

run = init_wandb(
    name=f"wgan_{DATASET}_001",
    config={
        "batch_size": 512,
        "epochs": 12000,
        "n_critic": 5,
        "clip_threshold": 0.01,
    }
)

# Configure 23 W&B charts
define_wgan_charts()
```

---

## Training with Per-Epoch Logging

```python
gan.train(
    x_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    run_folder=RUN_FOLDER,
    verbose=True,              # Detailed console output
    quality_metrics_every=100, # FID/IS every 100 epochs
    wandb_log=True             # W&B per-epoch logging
)
```

---

## Training Log Output

```
═══════════════════════════════════════════════════════════════════════════
Epoch 11999/12000 [1.46s]
───────────────────────────────────────────────────────────────────────────
  Losses     │ D: 5.4859 (R:5.4853 F:5.4865)  G: -115.549  W-dist: 115.55
  Weights    │ Critic μ:0.0024 σ:0.0096  Gen μ:0.0028 σ:0.0810
  Stability  │ D/G Ratio: 0.0475  Clip%: 0.0%  Var: 0.000002
═══════════════════════════════════════════════════════════════════════════

  Computing quality metrics...
  Quality    │ FID: 333.1  IS: 2.12±0.19  PixVar: 0.2960
```

---

## Metrics Reference

### Loss Metrics

| Symbol | Name | Formula | Interpretation |
|--------|------|---------|----------------|
| **D** | Critic Loss | `0.5 × (R + F)` | Combined critic loss |
| **R** | D_loss_real | `-E[f(x_real)]` | Critic on real images |
| **F** | D_loss_fake | `E[f(x_fake)]` | Critic on fake images |
| **G** | Generator Loss | `-E[f(G(z))]` | Generator objective |
| **W-dist** | Wasserstein Distance | `E[f(real)] - E[f(fake)]` | Earth Mover's Distance |

**Wasserstein Distance:**

$$W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \left( \mathbb{E}_{x \sim P_r}[f(x)] - \mathbb{E}_{z \sim P_z}[f(G(z))] \right)$$

**Healthy Signs:** D_real ≈ D_fake, W-dist increases then stabilizes, G loss more negative.

### Weight Statistics

| Symbol | Name | Interpretation |
|--------|------|----------------|
| **Critic μ** | Weight Mean | Should stay near 0 |
| **Critic σ** | Weight Std | Bounded by ±CLIP_THRESHOLD |
| **Gen μ** | Weight Mean | Should remain stable |
| **Gen σ** | Weight Std | Should stay > 0 |

### Stability Indicators

| Symbol | Name | Formula | Interpretation |
|--------|------|---------|----------------|
| **D/G Ratio** | Loss Ratio | `\|D\| / \|G\|` | Balance metric |
| **Clip%** | Clipping % | `(clipped/total) × 100` | Weight boundary hits |
| **Var** | Loss Variance | `var(losses)` | Training stability |

**Lipschitz Constraint:**

$$\|f\|_L = \sup_{x \neq y} \frac{|f(x) - f(y)|}{\|x - y\|} \leq K$$

**Clip% Interpretation:**

| Clip% | Stage | Meaning |
|-------|-------|---------|
| >20% | Early | Normal |
| Decreasing | Mid | Healthy |
| ~0% | Converged | Ideal |
| >50% | Persistent | Increase threshold |

**D/G Ratio:** >1.0 = critic strong, 0.05-0.1 = balanced, <0.01 = possible collapse.

### Quality Metrics (Every 100 Epochs)

| Symbol | Name | Interpretation |
|--------|------|----------------|
| **FID** | Fréchet Inception Distance | Lower = better |
| **IS** | Inception Score | Higher = better |
| **PixVar** | Pixel Variance | Diversity check |

**FID Formula:**

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2\sqrt{\Sigma_r \Sigma_g}\right)$$

| FID Range | Quality |
|-----------|---------|
| 0-50 | Excellent |
| 50-100 | Good |
| 100-200 | Moderate |
| 200-400 | Fair (CIFAR typical) |
| >400 | Poor |

**IS:** >4.0 = Good, 2.0-4.0 = Moderate, <2.0 = Limited

**PixVar:** 0.2-0.5 = Healthy, <0.1 = Mode collapse

---

## Mode Collapse Triage

### Symptoms

| Metric | Collapse | Healthy |
|--------|----------|---------|
| D Loss | 0.72-0.75 (stuck) | ~0.69-0.71 (fluctuating) |
| D Accuracy | 50% (stuck) | 50-70% (decreasing) |
| G Accuracy | 99%+ | 80-95% |

### Seed Control

Add before model building:

```python
import os, random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
```

### Triage Experiments

| Evidence | Root Cause | Solution |
|----------|------------|----------|
| All batch sizes work with seed 42, some fail with others | Random init | Use known-good seed |
| 256 works, 1024 fails across seeds | Batch size | Reduce batch or scale LR |
| Smaller batches more robust | Both | Use smaller batch or better init |

### Common Fixes

1. Reduce batch size
2. Lower learning rate
3. Increase n_critic
4. Reduce dropout
5. Use different seed

---

## Master Experiment Log Template

Track runs in notebook:

```markdown
| Run | Date | W&B URL | Batch | Epochs | Stability | D Loss | G Loss | Notes |
|-----|------|---------|-------|--------|-----------|--------|--------|-------|
| 001 | 2026-01-07 | [View](url) | 512 | 6000 | ✅ Stable | 5.35 | -116.4 | Baseline |
```

---

## Related Documentation

- **[TRAINING_STABILITY_ANALYSIS_TEMPLATE.md](TRAINING_STABILITY_ANALYSIS_TEMPLATE.md)** - Analysis report template
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - General training optimization
- **[NOTEBOOK_STANDARDIZATION.md](NOTEBOOK_STANDARDIZATION.md)** - Complete workflow
