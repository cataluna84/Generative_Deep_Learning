import json
import os

NOTEBOOK_PATH = 'v1/notebooks/03_01_autoencoder_train.ipynb'

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    
    # 1. Initialize BATCH_SIZE in Global Config
    # We look for the cell containing 'STATIC CONFIGURATION' or 'Global Configuration'
    # and add BATCH_SIZE = REFERENCE_BATCH_SIZE
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'REFERENCE_BATCH_SIZE =' in source and 'REFERENCE_EPOCHS =' in source:
                # Check if already added
                if 'BATCH_SIZE = REFERENCE_BATCH_SIZE' not in source:
                    # Add it after reference values
                    new_source = []
                    for line in cell['source']:
                        new_source.append(line)
                        if 'REFERENCE_EPOCHS =' in line:
                            new_source.append('\n')
                            new_source.append('# Initialize BATCH_SIZE (will be optimized later)\n')
                            new_source.append('BATCH_SIZE = REFERENCE_BATCH_SIZE\n')
                    cell['source'] = new_source
                    print("Added BATCH_SIZE initialization to Global Config.")
                break

    # 2. Remove "Dynamic Batch/Epoch Scaling" cell
    # Use a loop that allows modification of the list (iterate backwards or create new list)
    # We identify it by header "DYNAMIC BATCH SIZE & EPOCH SCALING"
    cells_to_keep = []
    skip_next_markdown = False
    
    for i, cell in enumerate(cells):
        should_keep = True
        
        if cell['cell_type'] == 'markdown':
            source = ''.join(cell['source'])
            if '## Dynamic Batch/Epoch Scaling' in source:
                print("Removing 'Dynamic Batch/Epoch Scaling' markdown header.")
                should_keep = False
        
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'DYNAMIC BATCH SIZE & EPOCH SCALING' in source:
                print("Removing 'Dynamic Batch/Epoch Scaling' code cell.")
                should_keep = False
        
        if should_keep:
            cells_to_keep.append(cell)
            
    notebook['cells'] = cells_to_keep
    cells = notebook['cells'] # Update reference

    # 3. Insert new optimization cell after Model Architecture
    # Logic: Find cell defining 'AE = Autoencoder('
    insert_index = -1
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'AE = Autoencoder(' in source:
                insert_index = i + 1
                break
    
    if insert_index != -1:
        # Check if cell already exists to avoid dupes
        next_cell_source = ""
        if insert_index < len(cells):
             next_cell_source = ''.join(cells[insert_index]['source'])
        
        if 'OPTIMIZE BATCH SIZE' not in next_cell_source:
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "# OPTIMIZE BATCH SIZE\n",
                    "# Use binary search to find the maximum batch size that fits in GPU memory\n",
                    "# ═══════════════════════════════════════════════════════════════════════════════\n",
                    "\n",
                    "if MODE == 'build':  # Only optimize if building new model\n",
                    "    print(\"\\nFinding optimal batch size...\")\n",
                    "    BATCH_SIZE = find_optimal_batch_size(AE.model, INPUT_DIM)\n",
                    "    EPOCHS = calculate_adjusted_epochs(REFERENCE_EPOCHS, REFERENCE_BATCH_SIZE, BATCH_SIZE)\n",
                    "    \n",
                    "    # Ensure minimum epochs\n",
                    "    EPOCHS = max(EPOCHS, 100)\n",
                    "\n",
                    "    # Update W&B config\n",
                    "    if wandb.run is not None:\n",
                    "        wandb.config.update({\n",
                    "            \"batch_size\": BATCH_SIZE,\n",
                    "            \"epochs\": EPOCHS\n",
                    "        }, allow_val_change=True)\n",
                    "\n",
                    "    print(f\"\\nFinal Batch Size: {BATCH_SIZE}\")\n",
                    "    print(f\"Final Epochs:     {EPOCHS}\")"
                ]
            }
            cells.insert(insert_index, new_cell)
            print("Inserted 'Optimize Batch Size' cell.")
        else:
            print("'Optimize Batch Size' cell already exists.")

    # Save
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook updated successfully.")

if __name__ == '__main__':
    update_notebook()
