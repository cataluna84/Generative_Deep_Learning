#!/usr/bin/env python3
"""
Script to modify GAN Camel notebook for run folder standardization.

Updates:
1. Imports - add 're' module
2. Configuration - Replace RUN_ID with DATASET_RUN_ID/EXPERIMENT_RUN_ID
3. Folder creation - Two-level structure with auto-increment
4. gan.train() call - Add save_weights_every/save_images_every
5. W&B init - New naming format
6. Experiment log - Replace Triage with Master log
"""
import json
import sys


def load_notebook(path):
    """Load notebook as JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_notebook(notebook, path):
    """Save notebook as JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)


def find_cell_containing(notebook, text):
    """Find index of cell containing specified text."""
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if text in source:
            return i
    return -1


def modify_imports_cell(notebook):
    """Add 're' import to imports cell."""
    idx = find_cell_containing(notebook, "import os")
    if idx == -1:
        print("ERROR: Could not find imports cell")
        return False
    
    cell = notebook['cells'][idx]
    source = cell['source']
    
    # Check if 're' already imported
    if any("import re" in line for line in source):
        print("  ⊘ 're' already imported")
        return True
    
    # Add 're' after 'import os'
    new_source = []
    for line in source:
        new_source.append(line)
        if "import os\n" in line:
            new_source.append("import re\n")
    
    cell['source'] = new_source
    print("  ✓ Added 'import re'")
    return True


def modify_config_cell(notebook):
    """Update configuration cell with DATASET_RUN_ID/EXPERIMENT_RUN_ID."""
    idx = find_cell_containing(notebook, "RUN_ID = '0001'")
    if idx == -1:
        print("ERROR: Could not find config cell with RUN_ID")
        return False
    
    new_config = '''# ═══════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION - PART 1: STATIC PARAMETERS
# ═══════════════════════════════════════════════════════════════════
# These values define the experiment identity and reference baselines.
# They do not depend on GPU detection or model building.

# -----------------------------------------------------------------------------
# Run Folder Configuration
# -----------------------------------------------------------------------------
SECTION = 'gan'
DATASET_RUN_ID = '0001'
DATA_NAME = 'camel'
EXPERIMENT_RUN_ID = None  # Set to None for auto-increment, or '013', etc.

# Base folder for this dataset
BASE_RUN_FOLDER = f'../run/{SECTION}/{DATASET_RUN_ID}_{DATA_NAME}'


def get_next_experiment_run_id(base_folder):
    """Determine next available experiment run ID."""
    if not os.path.exists(base_folder):
        return '001'
    existing_ids = []
    for item in os.listdir(base_folder):
        if os.path.isdir(os.path.join(base_folder, item)) and re.match(r'^\\d{3}$', item):
            existing_ids.append(int(item))
    if not existing_ids:
        return '001'
    return f'{max(existing_ids) + 1:03d}'


# Auto-generate EXPERIMENT_RUN_ID if not specified
if EXPERIMENT_RUN_ID is None or EXPERIMENT_RUN_ID == '':
    EXPERIMENT_RUN_ID = get_next_experiment_run_id(BASE_RUN_FOLDER)
    print(f'Auto-generated EXPERIMENT_RUN_ID: {EXPERIMENT_RUN_ID}')

# Full run folder path
RUN_FOLDER = f'{BASE_RUN_FOLDER}/{EXPERIMENT_RUN_ID}'

# Create run directories if they don't exist
if not os.path.exists(RUN_FOLDER):
    os.makedirs(RUN_FOLDER)
    os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
    os.makedirs(os.path.join(RUN_FOLDER, 'images'))
    os.makedirs(os.path.join(RUN_FOLDER, 'weights'))
    print(f'✓ Created run folder: {RUN_FOLDER}')
else:
    existing_files = os.listdir(RUN_FOLDER)
    if existing_files:
        print(f'⚠ WARNING: RUN_FOLDER exists with {len(existing_files)} items')

# -----------------------------------------------------------------------------
# Reference Training Configuration
# These are the original notebook values used for epoch scaling
# -----------------------------------------------------------------------------
REFERENCE_BATCH_SIZE = 256
REFERENCE_EPOCHS = 6000
MODE = 'build'  # Options: 'build' (new training), 'load' (resume)

print(f"Run folder: {RUN_FOLDER}")
print(f"Reference batch size: {REFERENCE_BATCH_SIZE}")
print(f"Reference epochs: {REFERENCE_EPOCHS}")'''
    
    notebook['cells'][idx]['source'] = [line + '\n' for line in new_config.split('\n')]
    # Remove trailing newline from last line
    notebook['cells'][idx]['source'][-1] = notebook['cells'][idx]['source'][-1].rstrip('\n')
    notebook['cells'][idx]['outputs'] = []  # Clear outputs
    
    print("  ✓ Updated configuration cell")
    return True


def modify_wandb_init_cell(notebook):
    """Update W&B initialization with new naming format."""
    idx = find_cell_containing(notebook, "wandb.init(")
    if idx == -1:
        print("ERROR: Could not find W&B init cell")
        return False
    
    cell = notebook['cells'][idx]
    source = ''.join(cell['source'])
    
    # Replace the name parameter
    # Current format likely: name=f"gan-camel-bs{BATCH_SIZE}-seed{SEED}"
    # New format: name=f"gan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}"
    
    import re as regex
    new_source = regex.sub(
        r'name=f"[^"]*"',
        'name=f"gan_{DATA_NAME}_{DATASET_RUN_ID}_{EXPERIMENT_RUN_ID}"',
        source
    )
    
    # Also add dataset_run_id and experiment_run_id to config
    if "dataset_run_id" not in new_source:
        # Add after "dataset": DATA_NAME,
        new_source = new_source.replace(
            '"dataset": DATA_NAME,',
            '"dataset": DATA_NAME,\n        "dataset_run_id": DATASET_RUN_ID,\n        "experiment_run_id": EXPERIMENT_RUN_ID,'
        )
    
    notebook['cells'][idx]['source'] = [new_source]
    print("  ✓ Updated W&B initialization")
    return True


def modify_train_call(notebook):
    """Add save_weights_every and save_images_every to gan.train() call."""
    idx = find_cell_containing(notebook, "gan.train(")
    if idx == -1:
        print("ERROR: Could not find gan.train() cell")
        return False
    
    cell = notebook['cells'][idx]
    source = ''.join(cell['source'])
    
    # Check if already has save_weights_every
    if "save_weights_every" in source:
        print("  ⊘ save_weights_every already present")
        return True
    
    # Add before the closing parenthesis
    # Find the last parameter line and add new ones
    source = source.replace(
        "lr_decay_epochs=LR_DECAY_EPOCHS\n)",
        "lr_decay_epochs=LR_DECAY_EPOCHS,\n    save_weights_every=500,\n    save_images_every=500\n)"
    )
    
    notebook['cells'][idx]['source'] = [source]
    print("  ✓ Updated gan.train() call")
    return True


def add_master_experiment_log(notebook):
    """Replace Triage Experiment Log with Master Experiment Log."""
    idx = find_cell_containing(notebook, "Triage Experiment Log")
    if idx == -1:
        print("ERROR: Could not find Triage Experiment Log cell")
        return False
    
    master_log = '''## Master Experiment Log

### Camel Dataset Experiments (DATASET_RUN_ID: 0001)

| Run | Date | W&B | Seed | Batch | Epochs | Dropout | Peak D Acc | Status | Notes |
|-----|------|-----|------|-------|--------|---------|------------|--------|-------|
| REF | - | [uexhdo3y](https://wandb.ai/cataluna84/generative-deep-learning/runs/uexhdo3y) | - | 1024 | - | 0.4 | - | 📌 Reference | GitHub baseline |
| - | - | - | 42 | 1024 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 123 | 1024 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 404 | 1024 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 707 | 1024 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 1024 | 1024 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 42 | 512 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | - | 1024 | 512 | 1500 | 0.2 | ~56% | ❌ Collapse | Phase 1-2 |
| - | - | [6b27357n](https://wandb.ai/cataluna84/generative-deep-learning/runs/6b27357n) | 404 | 512 | 1500 | 0.4 | 72.2% | ✅ Stable | Phase 3 |
| - | - | [07vsq1ga](https://wandb.ai/cataluna84/generative-deep-learning/runs/07vsq1ga) | 1024 | 512 | 1500 | 0.4 | 91.6% | ✅ Stable | Phase 3, Best |
| - | - | - | 42 | 512 | 1500 | 0.4 | 69.6% | ✅ Stable | Phase 3 |
| 012 | 2026-01-06 | [brzdilqv](https://wandb.ai/cataluna84/generative-deep-learning/runs/brzdilqv) | 707 | 512 | 1500 | 0.4 | 62.5% | ✅ Stable | Phase 3 |

---

### 🔬 Key Findings
1. **dropout=0.4 is critical** - all 4 seeds successful
2. **dropout=0.2 causes collapse** - all seeds failed
3. **Best seed: 1024** (91.6% peak D Acc)
4. **Stable convergence confirmed**: D loss ~0.736 (near optimal -log(0.5)≈0.693), balanced R/F losses
5. **No mode collapse**: Generator beats discriminator without causing collapse'''
    
    notebook['cells'][idx]['source'] = [line + '\n' for line in master_log.split('\n')]
    notebook['cells'][idx]['source'][-1] = notebook['cells'][idx]['source'][-1].rstrip('\n')
    
    print("  ✓ Replaced with Master Experiment Log")
    return True


def main():
    notebook_path = "v1/notebooks/04_01_gan_camel_train.ipynb"
    
    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)
    
    print("\nApplying modifications:")
    
    success = True
    success = modify_imports_cell(notebook) and success
    success = modify_config_cell(notebook) and success
    success = modify_wandb_init_cell(notebook) and success
    success = modify_train_call(notebook) and success
    success = add_master_experiment_log(notebook) and success
    
    if success:
        save_notebook(notebook, notebook_path)
        print(f"\n✓ Notebook saved: {notebook_path}")
    else:
        print("\n✗ Some modifications failed, notebook not saved")
        sys.exit(1)


if __name__ == "__main__":
    main()
