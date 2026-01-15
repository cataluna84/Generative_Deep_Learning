
import nbformat
from nbformat.v4 import new_markdown_cell
import os

NOTEBOOK_PATH = "v1/notebooks/05_01_cyclegan_train.ipynb"

def main():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        return

    print(f"Reading notebook: {NOTEBOOK_PATH}")
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # Content to insert
    instruction_text = """
## Downloading Data
Before running this notebook, you need to download the dataset. You can use the provided script:

```bash
bash ../../data_download_scripts/download_cyclegan_data.sh <dataset_name>
```

For example, to download the `apple2orange` dataset:
```bash
bash ../../data_download_scripts/download_cyclegan_data.sh apple2orange
```

Available datasets include: `apple2orange`, `summer2winter_yosemite`, `horse2zebra`, `monet2photo`, `cezanne2photo`, `ukiyoe2photo`, `vangogh2photo`, `maps`, `cityscapes`, `facades`, `iphone2dslr_flower`, `ae_photos`.
"""

    # check if cell already exists to avoid duplicates
    # simple check: look for "Downloading Data" in source
    for cell in nb.cells:
        if cell.cell_type == 'markdown' and "## Downloading Data" in cell.source:
            print("Instruction cell already exists. Skipping insertion.")
            return

    # Insert after the first cell (title cell)
    # The first cell is usually index 0. We insert at index 1.
    print("Inserting instruction cell at index 1...")
    new_cell = new_markdown_cell(instruction_text)
    nb.cells.insert(1, new_cell)

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    
    print("Notebook updated successfully.")

if __name__ == "__main__":
    main()
