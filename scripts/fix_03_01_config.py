import json
import os

NOTEBOOK_PATH = 'v1/notebooks/03_01_autoencoder_train.ipynb'

def fix_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    with open(NOTEBOOK_PATH, 'r') as f:
        notebook = json.load(f)

    cells = notebook['cells']
    
    # 1. Remove print_training_config from "Directory Setup" cell
    # Identify by looking for 'print_training_config(' in code cells near 'CREATE RUN DIRECTORIES'
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'CREATE RUN DIRECTORIES' in source and 'print_training_config' in source:
                # Remove lines related to print_training_config
                new_source = []
                # Simple logic: Keep lines until 'Print configuration summary' or 'print_training_config'
                skip = False
                for line in cell['source']:
                    if 'Print configuration summary' in line or 'print_training_config' in line:
                        skip = True
                    if not skip:
                        new_source.append(line)
                    if ')' in line and skip: # End of function call
                        skip = False # But actually we want to remove the block, so maybe just don't append
                        
                # Alternative: Just keep lines that started before the print block
                # The print block is at the end.
                cleaned_source = []
                for line in cell['source']:
                    if 'print_training_config' in line or 'vram_gb=GPU_VRAM_GB' in line or 'reference_batch=' in line or 'reference_epochs=' in line or 'Print configuration summary' in line:
                        continue
                    # Also handle the closing paren if on own line?
                    if line.strip() == ')' and len(cleaned_source) > 0 and 'print_training_config' not in ''.join(cleaned_source):
                        # This might be risky if other calls exist.
                        pass 
                    elif line.strip() == ')': # Likely the closing paren of the removed call
                        continue
                    else:
                        cleaned_source.append(line)
                
                cell['source'] = cleaned_source
                print("Removed print_training_config from Directory Setup.")

    # 2. Add print_training_config to "OPTIMIZE BATCH SIZE" cell
    # And ensure we get VRAM there
    for cell in cells:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'OPTIMIZE BATCH SIZE' in source:
                # Replace the simple prints at the end with print_training_config
                # We need to know where to insert.
                # Use split/join approach for safety
                
                # First, check if we need to add get_gpu_vram_gb call
                if 'get_gpu_vram_gb()' not in source:
                     # Insert it at start of build block
                     # Actually print_training_config can take vram_gb=get_gpu_vram_gb() call directly
                     pass

                new_lines = []
                source_lines = cell['source']
                
                # We want to replace the last few print lines
                for line in source_lines:
                    if 'print(f"\\nFinal Batch Size:' in line or 'print(f"Final Epochs:' in line:
                        continue
                    new_lines.append(line)
                
                # Add the new block
                code_to_add = [
                    "\n",
                    "    # Print final configuration\n",
                    "    print_training_config(\n",
                    "        BATCH_SIZE, EPOCHS,\n",
                    "        reference_batch=REFERENCE_BATCH_SIZE,\n",
                    "        reference_epochs=REFERENCE_EPOCHS,\n",
                    "        vram_gb=get_gpu_vram_gb()\n",
                    "    )\n"
                ]
                new_lines.extend(code_to_add)
                cell['source'] = new_lines
                print("Added print_training_config to Optimization cell.")

    # Save
    with open(NOTEBOOK_PATH, 'w') as f:
        json.dump(notebook, f, indent=1)
    print("Notebook fixed successfully.")

if __name__ == '__main__':
    fix_notebook()
