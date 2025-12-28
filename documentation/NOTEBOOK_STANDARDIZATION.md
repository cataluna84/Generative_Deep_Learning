# Notebook Standardization Workflow

This guide details the standard workflow for updating V1/V2 notebooks to include robust configuration, W&B tracking, and automatic learning rate optimization.

## Goal
Transform hardcoded, static notebooks into flexible, tracked, and optimized experiments.

## Recursive Update Steps

For each notebook:

### 1. Extract Global Layout
Move hardcoded parameters from the bottom of the notebook (training loop) to the top (after imports).

```python
# [New Cell] Global Configuration
BATCH_SIZE = 64  # or 1024 depending on VRAM
EPOCHS = 10
OPTIMIZER_NAME = 'adam'
DATASET_NAME = 'cifar10' # or whatever is appropriate
MODEL_TYPE = 'cnn'
LAYERS_DESC = '4_conv_2_dense' # descriptive string
```

### 2. Configure W&B
Initialize W&B early, but mark `learning_rate` as "auto" since we'll find it later.

```python
# [Modify W&B Init Cell]
run = init_wandb(
    name="02_03_cnn",
    project="generative-deep-learning",
    config={
        "model": MODEL_TYPE,
        "dataset": DATASET_NAME,
        "layers": LAYERS_DESC,
        "learning_rate": "auto",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER_NAME,
    }
)
```

### 3. Insert LRFinder Block
Before the main model training, insert the LRFinder workflow. **Crucially**, clone the model to avoid pre-training the main model weights.

```python
# [New Cell] Learning Rate Finder
lr_model = tf.keras.models.clone_model(model) # Must be defined before this cell
lr_opt = Adam(learning_rate=1e-6)
lr_model.compile(loss='categorical_crossentropy', optimizer=lr_opt, metrics=['accuracy'])

lr_finder = LRFinder(min_lr=1e-6, max_lr=1e-1, steps=100)
lr_model.fit(x_train, y_train,
             batch_size=BATCH_SIZE,
             steps_per_epoch=50,
             epochs=2,
             callbacks=[lr_finder],
             verbose=0)

lr_finder.plot_loss()
optimal_lr = lr_finder.get_optimal_lr()
print(f"Optimal LR: {optimal_lr}")

# Update W&B
wandb.config.update({"learning_rate": optimal_lr})
```

### 4. Update Main Training
Use the detected `optimal_lr` and global constants.

```python
# [Modify Optimizer Cell]
opt = Adam(learning_rate=optimal_lr)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
```

```python
# [Modify Fit Cell]
model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    # ... other params ...
    callbacks=[get_metrics_logger(), get_lr_scheduler()]
)
```
