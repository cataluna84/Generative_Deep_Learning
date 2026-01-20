#!/usr/bin/env python3
"""Update the POST-TRAINING VISUALIZATION cell to save to viz folder and log to W&B."""

import json
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).parent.parent / "v1/notebooks/06_01_lstm_text_train.ipynb"

# New POST-TRAINING VISUALIZATION code
NEW_VIZ_CODE = '''# ═══════════════════════════════════════════════════════════════════════════════
# POST-TRAINING VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt

if TRAIN_MODEL and 'history' in dir():
    history_dict = history.history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    axes[0].plot(history_dict['loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Over Epochs')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate
    if 'learning_rate' in history_dict:
        axes[1].semilogy(history_dict['learning_rate'], 'r-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate (log scale)')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, which='both', alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
        axes[1].set_title('Learning Rate (Not Available)')
    
    plt.tight_layout()
    
    # Save to viz folder
    viz_path = os.path.join(RUN_FOLDER, 'viz', 'training_history.png')
    plt.savefig(viz_path, dpi=150)
    
    # Log to W&B main dashboard
    wandb.log({"training_history": wandb.Image(fig)})
    
    plt.show()
    
    # Print summary
    print(f"\\n{'='*50}")
    print("TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"  Initial Loss  : {history_dict['loss'][0]:.6f}")
    print(f"  Final Loss    : {history_dict['loss'][-1]:.6f}")
    print(f"  Min Loss      : {min(history_dict['loss']):.6f} (Epoch {history_dict['loss'].index(min(history_dict['loss'])) + 1})")
    print(f"  Total Epochs  : {len(history_dict['loss'])}")
    if 'learning_rate' in history_dict:
        print(f"  Final LR      : {history_dict['learning_rate'][-1]:.2e}")
    print(f"{'='*50}")'''

# Updated folder structure doc
NEW_FOLDER_DOC = '''## Run Folder Configuration

Each training run is saved to a unique folder for experiment isolation. The folder structure is:

```
v1/run/write/0001_aesop/
├── 001/           # Experiment run 001
│   ├── viz/
│   │   └── training_history.png
│   └── weights/
│       ├── model.keras
│       ├── weights.weights.h5
│       └── tokenizer.json
├── 002/           # Experiment run 002
└── ...
```

### Auto-Increment Logic
If `EXPERIMENT_RUN_ID` is set to `None`, the system automatically finds the next available run number (001, 002, 003, etc.). This prevents accidentally overwriting previous experiments.'''

def update_notebook():
    # Load the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated_viz = False
    updated_doc = False
    
    for cell in notebook['cells']:
        # Update POST-TRAINING VISUALIZATION cell
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if '# POST-TRAINING VISUALIZATION' in source and 'plt.savefig' in source:
                # Replace with new code
                cell['source'] = [line + '\n' if i < len(NEW_VIZ_CODE.split('\n')) - 1 else line 
                                  for i, line in enumerate(NEW_VIZ_CODE.split('\n'))]
                updated_viz = True
                print("Updated POST-TRAINING VISUALIZATION cell")
        
        # Update folder structure documentation
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## Run Folder Configuration' in source and 'training_history.png' in source:
                cell['source'] = [line + '\n' if i < len(NEW_FOLDER_DOC.split('\n')) - 1 else line 
                                  for i, line in enumerate(NEW_FOLDER_DOC.split('\n'))]
                updated_doc = True
                print("Updated folder structure documentation")
    
    if updated_viz:
        # Save the notebook
        with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Saved notebook to: {NOTEBOOK_PATH}")
    else:
        print("Warning: Could not find POST-TRAINING VISUALIZATION cell to update")

if __name__ == "__main__":
    update_notebook()
