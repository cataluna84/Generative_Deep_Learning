# Brainstorm: Additional GAN Training Metrics

## Current Metrics Logged

From `WGAN.train()`:
- `d_loss` (total), `d_loss_real`, `d_loss_fake`
- `g_loss`

---

## Proposed Additional Metrics

### 🟢 Easy to Implement (Low Overhead)

| Metric | What it Shows | Plot Type |
|--------|---------------|-----------|
| **Wasserstein Distance** | `abs(g_loss)` - quality proxy | Line |
| **D-G Loss Ratio** | `d_loss / abs(g_loss)` - balance indicator | Line |
| **Loss Variance (rolling)** | Training stability | Line (smoothed) |
| **Epoch Time** | Training speed | Line |
| **Learning Rate** | LR schedule progress | Line |

---

### 🟡 Medium Effort (Weight/Gradient Analysis)

| Metric | What it Shows | Plot Type |
|--------|---------------|-----------|
| **Critic Gradient Norm** | Lipschitz constraint | Line |
| **Generator Gradient Norm** | Learning signal strength | Line |
| **Weight Clip Ratio** | % of weights clipped to threshold | Line |
| **Critic Weight Stats** | Mean/std of weights per layer | Histogram/Line |
| **Generator Weight Stats** | Mean/std of weights per layer | Histogram/Line |

**Implementation**: Compute during `train_critic()` / `train_generator()`

---

### 🔴 Higher Effort (Quality Metrics)

| Metric | What it Shows | Plot Type | Frequency |
|--------|---------------|-----------|-----------|
| **FID Score** | Image quality vs real | Line | Every N epochs |
| **Inception Score** | Quality + diversity | Line | Every N epochs |
| **Pixel Variance** | Output diversity | Line | Every epoch |
| **Generated Image Stats** | Mean/std of pixels | Histogram | Periodic |

**Note**: FID/IS require pre-trained Inception model (heavy computation)

---

## Recommended Priority

1. **Phase 1** (Minimal change):
   - Wasserstein distance (already have g_loss)
   - Epoch timing
   - Loss variance (rolling window)

2. **Phase 2** (Gradient monitoring):
   - Critic gradient norm
   - Generator gradient norm
   - Weight clip ratio

3. **Phase 3** (Quality metrics - optional):
   - FID score (every 100-500 epochs)
   - Pixel variance for diversity

---

## Visualization Ideas

### Enhanced Console Output
```
Epoch 100 [D: 0.035 (R:0.034 F:0.036)] [G: -5.23] [W-dist: 5.23] [∇C: 0.98] [∇G: 0.45] [1.2s]
```

### W&B Plots
1. **Losses Panel**: D loss, G loss, Wasserstein distance (3 lines)
2. **Gradients Panel**: Critic norm, Generator norm (2 lines)
3. **Stability Panel**: Loss variance, D-G ratio (2 lines)
4. **Weights Panel**: Weight distributions (histograms)

---

## Questions for User

1. **Priority**: Which metrics are most important to you?
   - [ ] Gradient norms
   - [ ] Weight statistics
   - [ ] FID/IS (expensive)
   - [ ] Timing info

2. **Console output format**: Compact (current) or verbose (more metrics)?

3. **FID frequency**: If implementing, how often? (e.g., every 500 epochs)
