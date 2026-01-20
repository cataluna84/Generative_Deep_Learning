#!/usr/bin/env python3
"""Update the on_epoch_end callback to only run every N epochs.

This reduces the overhead of text generation during training.
"""

import json

NOTEBOOK_PATH = "v1/notebooks/06_01_lstm_text_train.ipynb"

OLD_ON_EPOCH_END = '''def on_epoch_end(epoch, logs):
    """Generate sample text after each epoch for qualitative evaluation."""
    seed_text = ""
    gen_words = 500

    print('Temp 0.2')
    print(generate_text(seed_text, gen_words, model, SEQ_LENGTH, temp=0.2))
    print('Temp 0.33')
    print(generate_text(seed_text, gen_words, model, SEQ_LENGTH, temp=0.33))
    print('Temp 0.5')
    print(generate_text(seed_text, gen_words, model, SEQ_LENGTH, temp=0.5))
    print('Temp 1.0')
    print(generate_text(seed_text, gen_words, model, SEQ_LENGTH, temp=1))'''

NEW_ON_EPOCH_END = '''# Sample generation interval - only generate text every N epochs to reduce overhead
SAMPLE_EVERY_N_EPOCHS = 50  # Generate sample text every 50 epochs

def on_epoch_end(epoch, logs):
    """Generate sample text every N epochs for qualitative evaluation.
    
    This reduces training overhead while still providing periodic quality checks.
    Samples are generated at epochs: 0, 50, 100, 150, etc.
    """
    # Only generate samples at specified intervals (and epoch 0 for baseline)
    if epoch % SAMPLE_EVERY_N_EPOCHS != 0:
        return
    
    print(f"\\n{'='*60}")
    print(f"SAMPLE GENERATION - Epoch {epoch}")
    print(f"{'='*60}")
    
    seed_text = ""
    gen_words = 200  # Reduced from 500 for faster generation
    
    for temp in [0.2, 0.5, 1.0]:  # Reduced from 4 temperatures to 3
        print(f"\\n--- Temperature {temp} ---")
        print(generate_text(seed_text, gen_words, model, SEQ_LENGTH, temp=temp))'''


def update_callback():
    """Update the on_epoch_end callback in the notebook."""
    
    # Read the notebook
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Find and update the training cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'def on_epoch_end(epoch, logs):' in source and 'SAMPLE_EVERY_N_EPOCHS' not in source:
                # Replace the old callback with the new one
                new_source = source.replace(OLD_ON_EPOCH_END, NEW_ON_EPOCH_END)
                
                # Convert back to list format
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                # Remove trailing newline from last line
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                
                # Clear outputs
                cell['outputs'] = []
                cell['execution_count'] = None
                print("✓ Updated on_epoch_end callback to sample every 50 epochs")
                break
    
    # Also update EPOCHS to a more reasonable value
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'EPOCHS = 1000' in source:
                new_source = source.replace('EPOCHS = 1000', 'EPOCHS = 200  # Reduced from 1000; typically 50-200 is sufficient for this dataset')
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                cell['outputs'] = []
                cell['execution_count'] = None
                print("✓ Reduced EPOCHS from 1000 to 200")
                break
    
    # Write back
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"✓ Saved {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_callback()
