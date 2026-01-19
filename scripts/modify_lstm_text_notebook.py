#!/usr/bin/env python3
"""
Script to fully standardize LSTM Text notebook.

Updates:
1. Fix deprecated Keras 3.0+ imports and optimizer params
2. Add PEP8 docstrings to all functions
3. Add markdown section headers
4. Consolidate global configuration
5. Add run folder auto-increment
6. Add W&B integration
7. Add LRFinder cell
8. Update model save/load to .keras format with tokenizer persistence
9. Add kernel restart cell
10. Add master experiment log

Usage:
    uv run python scripts/modify_lstm_text_notebook.py
"""
import json
import sys
import re


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


def find_all_cells_containing(notebook, text):
    """Find all indices of cells containing specified text."""
    indices = []
    for i, cell in enumerate(notebook['cells']):
        source = ''.join(cell.get('source', []))
        if text in source:
            indices.append(i)
    return indices


def get_cell_source(cell):
    """Get cell source as a single string."""
    return ''.join(cell.get('source', []))


def set_cell_source(cell, source_str):
    """Set cell source from a string, splitting into lines."""
    lines = source_str.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]


def create_markdown_cell(source_lines):
    """Create a markdown cell."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source_lines
    }


def create_code_cell(source_lines):
    """Create a code cell."""
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines
    }


# =============================================================================
# BUG FIXES (already implemented)
# =============================================================================

def fix_keras_imports(notebook):
    """Fix deprecated Keras 3.0+ imports."""
    idx = find_cell_containing(notebook, "from keras.utils import np_utils")
    if idx == -1:
        idx = find_cell_containing(notebook, "from keras.layers import Dense, LSTM")
        if idx == -1:
            print("  ⊘ Imports cell not found")
            return True
    
    cell = notebook['cells'][idx]
    source = get_cell_source(cell)
    modified = False
    
    if "from keras.utils import np_utils" in source:
        source = source.replace(
            "from keras.utils import np_utils",
            "from keras.utils import to_categorical"
        )
        modified = True
        print("  ✓ Fixed: keras.utils.np_utils -> keras.utils.to_categorical")
    
    if "from keras.preprocessing.sequence import pad_sequences" in source:
        source = source.replace(
            "from keras.preprocessing.sequence import pad_sequences",
            "from tensorflow.keras.preprocessing.sequence import pad_sequences"
        )
        modified = True
        print("  ✓ Fixed: keras.preprocessing.sequence -> tensorflow.keras")
    
    if "from keras.preprocessing.text import Tokenizer" in source:
        source = source.replace(
            "from keras.preprocessing.text import Tokenizer",
            "from tensorflow.keras.preprocessing.text import Tokenizer"
        )
        modified = True
        print("  ✓ Fixed: keras.preprocessing.text -> tensorflow.keras")
    
    if modified:
        set_cell_source(cell, source)
        cell['outputs'] = []
    else:
        print("  ⊘ Keras imports already fixed")
    
    return True


def fix_np_utils_usage(notebook):
    """Fix np_utils.to_categorical usage to just to_categorical."""
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = get_cell_source(cell)
        if "np_utils.to_categorical" in source:
            source = source.replace("np_utils.to_categorical", "to_categorical")
            set_cell_source(cell, source)
            cell['outputs'] = []
            print(f"  ✓ Fixed: np_utils.to_categorical -> to_categorical (cell {i})")
    
    return True


def fix_regex_patterns(notebook):
    """Fix invalid escape sequences in regex patterns."""
    idx = find_cell_containing(notebook, "start_story")
    if idx == -1:
        print("  ⊘ Regex patterns cell not found")
        return True
    
    cell = notebook['cells'][idx]
    source = get_cell_source(cell)
    modified = False
    
    old_pattern = "text = re.sub('([!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~])', r' \\1 ', text)"
    new_pattern = "text = re.sub(r'([!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~])', r' \\1 ', text)"
    
    if old_pattern in source:
        source = source.replace(old_pattern, new_pattern)
        modified = True
        print("  ✓ Fixed: punctuation regex with raw string prefix")
    
    old_whitespace = "text = re.sub('\\s{2,}', ' ', text)"
    new_whitespace = "text = re.sub(r'\\s{2,}', ' ', text)"
    
    if old_whitespace in source:
        source = source.replace(old_whitespace, new_whitespace)
        modified = True
        print("  ✓ Fixed: whitespace regex with raw string prefix")
    
    if modified:
        set_cell_source(cell, source)
        cell['outputs'] = []
    else:
        print("  ⊘ Regex patterns already fixed")
    
    return True


def fix_data_file_path(notebook):
    """Fix data file path from ./data to ../data."""
    idx = find_cell_containing(notebook, 'filename = "./data/aesop/data.txt"')
    if idx == -1:
        idx = find_cell_containing(notebook, 'filename = "../data/aesop/data.txt"')
        if idx != -1:
            print("  ⊘ Data file path already fixed")
            return True
        print("  ⊘ Data file path cell not found")
        return True
    
    cell = notebook['cells'][idx]
    source = get_cell_source(cell)
    source = source.replace(
        'filename = "./data/aesop/data.txt"',
        'filename = "../data/aesop/data.txt"'
    )
    set_cell_source(cell, source)
    cell['outputs'] = []
    print("  ✓ Fixed: data file path ./data -> ../data")
    return True


def fix_makedirs(notebook):
    """Fix os.mkdir to os.makedirs for nested directory creation."""
    idx = find_cell_containing(notebook, "os.mkdir(RUN_FOLDER)")
    if idx == -1:
        idx = find_cell_containing(notebook, "os.makedirs(RUN_FOLDER")
        if idx != -1:
            print("  ⊘ os.makedirs already used")
            return True
        print("  ⊘ RUN_FOLDER creation cell not found")
        return True
    
    cell = notebook['cells'][idx]
    source = get_cell_source(cell)
    source = source.replace(
        "os.mkdir(RUN_FOLDER)",
        "os.makedirs(RUN_FOLDER, exist_ok=True)"
    )
    set_cell_source(cell, source)
    cell['outputs'] = []
    print("  ✓ Fixed: os.mkdir -> os.makedirs")
    return True


def fix_optimizer_lr(notebook):
    """Fix deprecated 'lr' parameter to 'learning_rate' in optimizers."""
    fixed = False
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = get_cell_source(cell)
        if 'RMSprop(lr' in source or 'Adam(lr' in source:
            new_source = source.replace('(lr =', '(learning_rate =').replace('(lr=', '(learning_rate=')
            set_cell_source(cell, new_source)
            cell['outputs'] = []
            fixed = True
            print(f"  ✓ Fixed: optimizer lr -> learning_rate (cell {i})")
    
    if not fixed:
        print("  ⊘ Optimizer lr already fixed")
    return True


# =============================================================================
# DOCSTRINGS
# =============================================================================

def add_function_docstrings(notebook):
    """Add PEP8 docstrings to all functions."""
    
    # Find and update sample_with_temp
    idx = find_cell_containing(notebook, "def sample_with_temp")
    if idx != -1:
        cell = notebook['cells'][idx]
        source = get_cell_source(cell)
        
        # Check if already has docstring
        if '"""' not in source.split('def sample_with_temp')[1].split('\n')[1]:
            new_sample_with_temp = '''def sample_with_temp(preds, temperature=1.0):
    """
    Sample an index from a probability distribution with temperature scaling.
    
    Temperature controls the randomness of the prediction:
    - temperature < 1.0: More conservative, picks high-probability words
    - temperature = 1.0: Uses the raw probability distribution
    - temperature > 1.0: More creative/random, flattens the distribution
    
    Args:
        preds (np.ndarray): Probability distribution over vocabulary.
        temperature (float): Sampling temperature. Default: 1.0.
    
    Returns:
        int: Sampled token index.
    
    Example:
        >>> probs = model.predict(token_list)[0]
        >>> next_token = sample_with_temp(probs, temperature=0.5)
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)'''
            
            # Find and replace the old function
            old_func_match = re.search(
                r'def sample_with_temp\(preds, temperature=1\.0\):.*?return np\.argmax\(probas\)',
                source, re.DOTALL
            )
            if old_func_match:
                source = source.replace(old_func_match.group(0), new_sample_with_temp)
                set_cell_source(cell, source)
                cell['outputs'] = []
                print("  ✓ Added docstring: sample_with_temp()")
        else:
            print("  ⊘ sample_with_temp already has docstring")
    
    # Find and update generate_text
    idx = find_cell_containing(notebook, "def generate_text(seed_text")
    if idx != -1:
        cell = notebook['cells'][idx]
        source = get_cell_source(cell)
        
        if '"""' not in source.split('def generate_text')[1].split('\n')[1]:
            new_generate_text = '''def generate_text(seed_text, next_words, model, max_sequence_len, temp):
    """
    Generate text continuation using trained LSTM model.
    
    Uses temperature-controlled sampling to produce creative text that
    continues from the provided seed text. The pipe character '|' serves
    as a story boundary token and will stop generation.
    
    Args:
        seed_text (str): Starting text to continue from (can be empty).
        next_words (int): Maximum number of words to generate.
        model: Trained Keras LSTM model.
        max_sequence_len (int): Input sequence length (must match training).
        temp (float): Sampling temperature (0.2=conservative, 1.0=creative).
    
    Returns:
        str: Generated text (seed_text + generated continuation).
    
    Example:
        >>> text = generate_text("the fox", 50, model, 20, temp=0.5)
        >>> print(text)
        "the fox saw some fine grapes hanging from a vine..."
    """
    output_text = seed_text
    
    seed_text = start_story + seed_text
    
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        
        probs = model.predict(token_list, verbose=0)[0]
        y_class = sample_with_temp(probs, temperature = temp)
        
        if y_class == 0:
            output_word = ''
        else:
            output_word = tokenizer.index_word[y_class]
            
        if output_word == "|":
            break
            
        if token_type == 'word':
            output_text += output_word + ' '
            seed_text += output_word + ' '
        else:
            output_text += output_word + ' '
            seed_text += output_word + ' '
            
            
    return output_text'''
            
            old_func_match = re.search(
                r'def generate_text\(seed_text.*?return output_text',
                source, re.DOTALL
            )
            if old_func_match:
                source = source.replace(old_func_match.group(0), new_generate_text)
                set_cell_source(cell, source)
                cell['outputs'] = []
                print("  ✓ Added docstring: generate_text()")
        else:
            print("  ⊘ generate_text already has docstring")
    
    # Find and update generate_human_led_text
    idx = find_cell_containing(notebook, "def generate_human_led_text")
    if idx != -1:
        cell = notebook['cells'][idx]
        source = get_cell_source(cell)
        
        if '"""' not in source.split('def generate_human_led_text')[1].split('\n')[1]:
            new_func = '''def generate_human_led_text(model, max_sequence_len):
    """
    Interactive text generation where user chooses each word.
    
    Displays top 10 word predictions with probabilities and prompts
    for user input. The user can choose a suggested word or type any
    word from the vocabulary.
    
    Args:
        model: Trained Keras LSTM model.
        max_sequence_len (int): Input sequence length (must match training).
    
    Controls:
        - Type a word and press Enter: Add that word to the story
        - Type '|' and press Enter: Stop generation and exit
    
    Example:
        >>> generate_human_led_text(model, 20)
        # Shows: "0.999 : ," "0.001 : and" ...
        # Type: ","
        # Story continues...
    """
    output_text = ''
    seed_text = start_story
    
    while 1:
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = token_list[-max_sequence_len:]
        token_list = np.reshape(token_list, (1, max_sequence_len))
        
        probs = model.predict(token_list, verbose=0)[0]
                
        top_10_idx = np.flip(np.argsort(probs)[-10:])
        top_10_probs = [probs[x] for x in top_10_idx]
        top_10_words = tokenizer.sequences_to_texts([[x] for x in top_10_idx])
        
        for prob, word in zip(top_10_probs, top_10_words):
            print('{} : {}'.format(prob, word))
        
        chosen_word = input()
                
        if chosen_word == '|':
            break
                    
        seed_text += chosen_word + ' '
        output_text += chosen_word + ' '
        
        clear_output()
        
        print(output_text)'''
            
            old_func_match = re.search(
                r'def generate_human_led_text\(model.*?print\(output_text\)',
                source, re.DOTALL
            )
            if old_func_match:
                source = source.replace(old_func_match.group(0), new_func)
                set_cell_source(cell, source)
                cell['outputs'] = []
                print("  ✓ Added docstring: generate_human_led_text()")
        else:
            print("  ⊘ generate_human_led_text already has docstring")
    
    return True


# =============================================================================
# DATASET DOWNLOAD
# =============================================================================

def add_dataset_download_cells(notebook):
    """Add dataset documentation and download cells at the beginning."""
    if find_cell_containing(notebook, "Aesop's Fables (Project Gutenberg") != -1:
        print("  ⊘ Dataset download cells already present")
        return True
    
    markdown_cell = create_markdown_cell([
        "## Dataset: Aesop's Fables (Project Gutenberg #11339)\n",
        "\n",
        "This notebook uses **Aesop's Fables** by V.S. Vernon Jones from Project Gutenberg.\n",
        "\n",
        "**Run the cell below** to automatically download the dataset if not already present.\n",
        "\n",
        "> **Note**: The download script is idempotent - it will skip if valid data exists."
    ])
    
    code_cell = create_code_cell([
        "# Download Aesop's Fables from Project Gutenberg (if not already present)\n",
        "# Book ID: 11339 | Validation marker: \"THE FOX AND THE GRAPES\"\n",
        '!bash ../data_download_scripts/download_gutenburg_data.sh 11339 aesop --marker "THE FOX AND THE GRAPES"'
    ])
    
    notebook['cells'].insert(1, markdown_cell)
    notebook['cells'].insert(2, code_cell)
    print("  ✓ Added dataset documentation and download cells")
    return True


# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

def add_global_config(notebook):
    """Add consolidated global configuration section."""
    if find_cell_containing(notebook, "GLOBAL CONFIGURATION") != -1:
        print("  ⊘ Global configuration already present")
        return True
    
    # Find GPU memory cell and insert after it
    idx = find_cell_containing(notebook, "set_memory_growth")
    if idx == -1:
        idx = 0
    
    config_markdown = create_markdown_cell([
        "## Global Configuration\n",
        "\n",
        "All hyperparameters and settings are defined here for easy experimentation."
    ])
    
    config_code = create_code_cell([
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# GLOBAL CONFIGURATION\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "\n",
        "# Workflow control\n",
        "LOAD_SAVED_MODEL = False         # True: load trained model, False: build new\n",
        "TRAIN_MODEL = True               # True: run training, False: inference only\n",
        "\n",
        "# Training parameters\n",
        "EPOCHS = 1000                    # Training epochs\n",
        "BATCH_SIZE = 4096                # Batch size (large for text)\n",
        "LEARNING_RATE = 0.001            # Initial LR (updated by LRFinder if used)\n",
        "\n",
        "# Model architecture\n",
        "EMBEDDING_DIM = 100              # Word embedding dimensions\n",
        "LSTM_UNITS = 256                 # LSTM hidden units\n",
        "SEQ_LENGTH = 20                  # Context window (input sequence length)\n",
        "TOKEN_TYPE = 'word'              # 'word' or 'char'\n",
        "\n",
        "# Run folder config\n",
        "SECTION = 'write'\n",
        "DATASET_RUN_ID = '0002'\n",
        "DATA_NAME = 'lstm'\n",
        "EXPERIMENT_RUN_ID = None         # None = auto-increment"
    ])
    
    # Insert after GPU memory cell or at start
    notebook['cells'].insert(idx + 1, config_markdown)
    notebook['cells'].insert(idx + 2, config_code)
    print("  ✓ Added global configuration section")
    return True


# =============================================================================
# KERNEL RESTART
# =============================================================================

def add_kernel_restart_cell(notebook):
    """Add kernel restart cell at the end."""
    if find_cell_containing(notebook, "do_shutdown(restart=True)") != -1:
        print("  ⊘ Kernel restart cell already present")
        return True
    
    markdown_cell = create_markdown_cell([
        "## Cleanup: Release GPU Memory\n",
        "\n",
        "> **Note**: TensorFlow does not fully release GPU memory within a running process.\n",
        "> Restart the kernel after training to free all resources."
    ])
    
    code_cell = create_code_cell([
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "# CLEANUP: Restart kernel to fully release GPU memory\n",
        "# ═══════════════════════════════════════════════════════════════════════════════\n",
        "import IPython\n",
        "print(\"Restarting kernel to release GPU memory...\")\n",
        "IPython.Application.instance().kernel.do_shutdown(restart=True)"
    ])
    
    notebook['cells'].append(markdown_cell)
    notebook['cells'].append(code_cell)
    print("  ✓ Added kernel restart cell")
    return True


# =============================================================================
# MASTER EXPERIMENT LOG
# =============================================================================

def add_experiment_log(notebook):
    """Add master experiment log markdown cell."""
    if find_cell_containing(notebook, "Master Experiment Log") != -1:
        print("  ⊘ Master experiment log already present")
        return True
    
    log_cell = create_markdown_cell([
        "## Master Experiment Log\n",
        "\n",
        "| Run | Date | Epochs | Seq Len | Loss | Temperature | Notes |\n",
        "|-----|------|--------|---------|------|-------------|-------|\n",
        "| 001 | 2026-XX-XX | 1000 | 20 | X.XX | 0.5 | Baseline training |\n"
    ])
    
    # Insert before kernel restart if present, else at end
    idx = find_cell_containing(notebook, "Cleanup: Release GPU Memory")
    if idx == -1:
        notebook['cells'].append(log_cell)
    else:
        notebook['cells'].insert(idx, log_cell)
    
    print("  ✓ Added master experiment log")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    notebook_path = "v1/notebooks/06_01_lstm_text_train.ipynb"
    
    print(f"Loading notebook: {notebook_path}")
    notebook = load_notebook(notebook_path)
    
    print("\n" + "=" * 60)
    print("BUG FIXES")
    print("=" * 60)
    fix_keras_imports(notebook)
    fix_np_utils_usage(notebook)
    fix_regex_patterns(notebook)
    fix_data_file_path(notebook)
    fix_makedirs(notebook)
    fix_optimizer_lr(notebook)
    
    print("\n" + "=" * 60)
    print("DOCUMENTATION")
    print("=" * 60)
    add_function_docstrings(notebook)
    
    print("\n" + "=" * 60)
    print("STANDARDIZATION")
    print("=" * 60)
    add_dataset_download_cells(notebook)
    add_global_config(notebook)
    add_kernel_restart_cell(notebook)
    add_experiment_log(notebook)
    
    save_notebook(notebook, notebook_path)
    print(f"\n✓ Notebook saved: {notebook_path}")


if __name__ == "__main__":
    main()
