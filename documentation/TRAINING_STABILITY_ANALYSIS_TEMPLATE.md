# Training Stability Analysis Template

This template documents training run analysis for any model type (VAE, GAN, Autoencoder, Deep Learning).
Save as `{EXPERIMENT_RUN_ID}_analysis_report.md` in your experiment run folder.

---

# Training Analysis Report: {EXPERIMENT_RUN_ID}

**Generated**: [DATE]  
**Model Type**: [MODEL_TYPE] (VAE / GAN / Autoencoder / CNN / etc.)  
**Run Folder**: `{RUN_FOLDER}`  
**Total Epochs**: [EPOCHS]  
**Final Loss**: [LOSS]  
**W&B Run**: [View on W&B](https://wandb.ai/USERNAME/PROJECT/runs/RUN_ID)

---

## Run Identification

| Parameter | Value |
|-----------|-------|
| Section | [SECTION] (e.g., `vae`, `gan`, `autoencoder`) |
| Dataset Run ID | [DATASET_RUN_ID] |
| Data Name | [DATA_NAME] |
| Experiment Run ID | [EXPERIMENT_RUN_ID] |
| Full Path | `v1/run/[SECTION]/[DATASET_RUN_ID]_[DATA_NAME]/[EXPERIMENT_RUN_ID]/` |

---

## Training Verdict

| Metric | Value |
|--------|-------|
| **Stability** | ✅ STABLE / ⚠️ UNSTABLE / ❌ FAILED |
| **Quality** | Excellent / Good / Fair / Poor |
| **Score** | X/Y indicators passed |
| **Recommendation** | [Action to take] |

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | [VALUE] |
| Epochs | [VALUE] |
| Learning Rate | [VALUE] |
| Optimizer | [VALUE] |
| Input Dim | [VALUE] |
| Latent Dim (Z) | [VALUE] |

### Model-Specific Parameters

#### For VAE/Autoencoder:
| Parameter | Value |
|-----------|-------|
| Encoder Filters | [VALUE] |
| Decoder Filters | [VALUE] |
| KL Weight (β) | [VALUE] |

#### For GAN:
| Parameter | Value |
|-----------|-------|
| LR Critic | [VALUE] |
| LR Generator | [VALUE] |
| N Critic | [VALUE] |
| Clip Threshold | [VALUE] |
| Critic Filters | [VALUE] |
| Generator Filters | [VALUE] |

---

## Training Progression (Phase-wise Metrics)

### Standard Models (VAE/Autoencoder/CNN)

| Phase | Epoch Range | Loss (Start → End) | Δ Loss/epoch |
|-------|-------------|--------------------|--------------| 
| Warmup | 0-10 | X.XX → X.XX | -X.XXXX |
| Early | 10-50 | X.XX → X.XX | -X.XXXX |
| Mid | 50-150 | X.XX → X.XX | -X.XXXX |
| Late | 150-TOTAL | X.XX → X.XX | -X.XXXX |

### GAN Models

| Phase | Epoch Range | D Loss (Start → End) | G Loss (Start → End) | Δ D/epoch | Δ G/epoch |
|-------|-------------|----------------------|----------------------|-----------|-----------|
| Warmup | 0-100 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Early | 100-1000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Mid | 1000-4000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Late | 4000-8000 | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |
| Final | 8000-TOTAL | X.XX → X.XX | -X.XX → -X.XX | X.XXXX | -X.XXXX |

---

## Stability Indicators

### Standard Models (VAE/Autoencoder/CNN)

| Indicator | Status | Observation |
|-----------|--------|-------------|
| Convergence | ✅/⚠️/❌ | Loss converging / plateauing / diverging |
| Smoothness | ✅/⚠️/❌ | Smooth training / minor oscillations / unstable |
| LR Schedule | ✅/⚠️/❌ | Appropriate reductions / too aggressive / not triggered |
| Early Stopping | ✅/⚠️/❌ | Not triggered / triggered early / triggered at limit |

### GAN Models

| Indicator | Status | Observation |
|-----------|--------|-------------|
| Monotonicity | ✅/⚠️/❌ | D and G losses change smoothly / with oscillations |
| Balance | ✅/⚠️/❌ | D_loss_real ≈ D_loss_fake (avg diff: X.XXX) |
| Mode Collapse | ✅/⚠️/❌ | No sudden plateaus / Possible collapse detected |
| Gradient Signal | ✅/⚠️/❌ | Critic maintains / loses discrimination ability |
| Wasserstein Distance | ✅/⚠️/❌ | |G loss| grows steadily (X.XX → X.XX) |

---

## Interpretation

### For VAE/Autoencoder:
- **Reconstruction Loss**: Measures output quality (lower = better)
- **KL Divergence**: Regularization forcing latent space to N(0,1)
- **Total Loss**: Weighted sum (R_loss + β × KL_loss)

### For GAN (WGAN):
- **D loss = E[critic(real)] - E[critic(fake)]**: Critic maximizes this
- **G loss = -E[critic(fake)]**: Generator minimizes this
- **Expected Behavior**: D loss positive/increasing, |G loss| increasing

---

## Quality Metrics

### VAE/Autoencoder:
| Metric | Value | Assessment |
|--------|-------|------------|
| MSE/BCE | [VALUE] | Excellent/Good/Fair/Poor |
| SSIM | [VALUE] | Image quality |
| Latent Coverage | [VALUE] | Representation quality |

### GAN:
| Metric | Value | Assessment |
|--------|-------|------------|
| FID | [VALUE] | Excellent/Good/Fair/Poor |
| IS | [VALUE] ± [STD] | Excellent/Moderate/Limited |
| PixVar | [VALUE] | Good diversity / Mode collapse risk |

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `{EXPERIMENT_RUN_ID}_analysis_report.md` | This report |
| `model.keras` | Full trained model |
| `weights/weights.weights.h5` | Model weights only |
| `images/*.png` | Generated samples |
| `viz/*.png` | Training visualizations |

---

## Notes

[Add observations, hypotheses, or next steps here]

---

## Full Details

For complete metrics, loss curves, and generated images, see the W&B run dashboard.

[View Full Report on W&B](https://wandb.ai/USERNAME/PROJECT/runs/RUN_ID)
