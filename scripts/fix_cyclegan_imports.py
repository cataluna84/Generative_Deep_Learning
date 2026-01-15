#!/usr/bin/env python3
"""Fix directory creation bug in 05_01_cyclegan_train.ipynb.

This script fixes the FileNotFoundError caused by os.mkdir() failing
when parent directories don't exist. Changes os.mkdir to os.makedirs.

The fix changes:
    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
        ...

To:
    if not os.path.exists(RUN_FOLDER):
        os.makedirs(RUN_FOLDER)
        os.makedirs(os.path.join(RUN_FOLDER, 'viz'))
        ...
"""

import json
from pathlib import Path


def fix_cyclegan_makedirs():
    """Fix directory creation in the CycleGAN training notebook."""
    notebook_path = Path(__file__).parent.parent / "v1" / "notebooks" / "05_01_cyclegan_train.ipynb"
    
    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return False
    
    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and fix the cell with the directory creation
    fixed = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell.get('source', [])
            source_str = ''.join(source) if isinstance(source, list) else source
            
            # Check if this cell has the directory creation with os.mkdir
            if 'RUN_FOLDER' in source_str and 'os.mkdir(RUN_FOLDER)' in source_str:
                # Replace os.mkdir with os.makedirs
                new_source = []
                for line in source:
                    # Replace os.mkdir with os.makedirs
                    if 'os.mkdir(' in line:
                        line = line.replace('os.mkdir(', 'os.makedirs(')
                    new_source.append(line)
                
                cell['source'] = new_source
                
                # Clear the error output from this cell
                cell['outputs'] = []
                cell['execution_count'] = None
                
                fixed = True
                print("Fixed directory creation by replacing os.mkdir with os.makedirs")
                print("This allows parent directories to be created automatically")
                break
    
    if not fixed:
        print("No cell needing fix found in notebook")
        return False
    
    # Write the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Successfully updated: {notebook_path}")
    return True


if __name__ == '__main__':
    fix_cyclegan_makedirs()
