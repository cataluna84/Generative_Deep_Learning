import json
import os

NOTEBOOK_PATH = 'v1/notebooks/03_01_autoencoder_train.ipynb'

def fix_epochs():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    
    # Locate Global Configuration cell
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'BATCH_SIZE = REFERENCE_BATCH_SIZE' in source:
                if 'EPOCHS = REFERENCE_EPOCHS' not in source:
                    new_source = []
                    for line in cell['source']:
                        new_source.append(line)
                        if 'BATCH_SIZE = REFERENCE_BATCH_SIZE' in line:
                            new_source.append('    EPOCHS = REFERENCE_EPOCHS\n')
                    
                    cell['source'] = new_source
                    print("Added EPOCHS initialization.")
                else:
                    print("EPOCHS already initialized.")

    # Save
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated.")

if __name__ == '__main__':
    fix_epochs()
