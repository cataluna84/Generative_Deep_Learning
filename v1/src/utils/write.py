#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Question-Answer Data Processing Utilities.

This module provides utilities for loading, processing, and preparing
QA (Question-Answer) training data from CSV files for seq2seq model training.

Features:
    - GloVe word embedding loading and vocabulary management
    - Tokenization and batch preparation for seq2seq models
    - Configurable batch size for different GPU memory sizes
    - Training and test data generators with shuffling

Dependencies:
    - numpy: For array operations
    - GloVe embeddings file: `data/glove/glove.6B.100d.trimmed.txt`

Note:
    This module loads trimmed GloVe embeddings at import time. Ensure the
    embeddings file exists by running `download_qa_data.sh` first.

Example:
    Basic usage for training::

        from src.utils.write import training_data, test_data, glove, BATCH_SIZE

        # Check vocabulary size
        print(f"Vocabulary size: {glove.shape[0]}")
        print(f"Embedding dimensions: {glove.shape[1]}")

        # Iterate over training batches with custom batch size
        for batch in training_data(batch_size=256):
            document_tokens = batch['document_tokens']
            answer_labels = batch['answer_labels']
            # ... train model ...

Attributes:
    glove (np.ndarray): Pre-loaded GloVe embeddings matrix of shape
        (vocab_size, embedding_dim).
    DEFAULT_BATCH_SIZE (int): Default batch size for data generators.
    PADDING_TOKEN (int): Token ID for padding (0).
    UNKNOWN_TOKEN (int): Token ID for unknown/OOV words (1).
    START_TOKEN (int): Token ID for sequence start marker (2).
    END_TOKEN (int): Token ID for sequence end marker (3).
"""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import Counter
from pathlib import Path
from typing import Any, Generator, Optional

import csv
import random

import numpy as np


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Batch size configuration
# Adjust based on GPU VRAM:
#   - 128: ~4.5 GB (Colab free tier)
#   - 256: ~6-7 GB
#   - 384: ~7.5 GB (8GB VRAM)
DEFAULT_BATCH_SIZE: int = 384

# Maximum document length (words beyond this are truncated)
_MAX_DOC_LENGTH: int = 200


# =============================================================================
# SPECIAL TOKENS
# =============================================================================

# Special token strings
PADDING_WORD: str = "<PAD>"
UNKNOWN_WORD: str = "<UNK>"
START_WORD: str = "<START>"
END_WORD: str = "<END>"


# =============================================================================
# VOCABULARY MANAGEMENT
# =============================================================================

# Internal vocabulary mappings (populated at module load time)
_word_to_idx: dict[str, int] = {}
_idx_to_word: list[str] = []


def _add_word(word: str) -> int:
    """Add a word to the vocabulary and return its token ID.

    This is an internal function used during vocabulary initialization.
    Words are assigned sequential integer IDs starting from 0.

    Args:
        word: The word string to add to vocabulary.

    Returns:
        The token ID assigned to the word.

    Note:
        This function does not check for duplicates. Adding the same word
        twice will create duplicate entries.
    """
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


# Initialize special tokens with reserved IDs (0-3)
PADDING_TOKEN: int = _add_word(PADDING_WORD)  # ID: 0
UNKNOWN_TOKEN: int = _add_word(UNKNOWN_WORD)  # ID: 1
START_TOKEN: int = _add_word(START_WORD)      # ID: 2
END_TOKEN: int = _add_word(END_WORD)          # ID: 3


# =============================================================================
# GLOVE EMBEDDINGS LOADING
# =============================================================================

# Calculate paths relative to this module file (v1/src/utils/write.py)
# Data directory is at v1/data/, which is 2 levels up from src/utils/
_module_dir = Path(__file__).parent  # v1/src/utils/
_embeddings_path = _module_dir / "../../data/glove/glove.6B.100d.trimmed.txt"

# Load GloVe embeddings at module import time
# This populates the vocabulary and embedding matrix
with open(str(_embeddings_path), encoding="utf-8") as _f:
    # Determine embedding dimensions from first line
    _first_line = _f.readline()
    _chunks = _first_line.split(" ")
    _dimensions = len(_chunks) - 1
    _f.seek(0)

    # Count vocabulary size (add 4 for special tokens)
    _vocab_size = sum(1 for _ in _f) + 4
    _f.seek(0)

    # Initialize embedding matrix
    glove: np.ndarray = np.ndarray((_vocab_size, _dimensions), dtype=np.float32)

    # Initialize special token embeddings with small random vectors
    # Using normal distribution with mean=0, std=0.02
    glove[PADDING_TOKEN] = np.random.normal(0, 0.02, _dimensions)
    glove[UNKNOWN_TOKEN] = np.random.normal(0, 0.02, _dimensions)
    glove[START_TOKEN] = np.random.normal(0, 0.02, _dimensions)
    glove[END_TOKEN] = np.random.normal(0, 0.02, _dimensions)

    # Load word embeddings from file
    for _line in _f:
        _chunks = _line.split(" ")
        _word = _chunks[0]
        _idx = _add_word(_word)
        glove[_idx] = [float(chunk) for chunk in _chunks[1:]]

        # Safety check to prevent array overflow
        if len(_idx_to_word) >= _vocab_size:
            break


# =============================================================================
# VOCABULARY LOOKUP FUNCTIONS
# =============================================================================

def look_up_word(word: str) -> int:
    """Convert a word to its token ID.

    Looks up a word in the vocabulary and returns its corresponding
    integer token ID. Unknown words return the UNKNOWN_TOKEN ID.

    Args:
        word: The word string to look up. Case-sensitive.

    Returns:
        Token ID for the word, or UNKNOWN_TOKEN (1) if not in vocabulary.

    Example:
        >>> look_up_word("the")
        42
        >>> look_up_word("xyzabc123")  # Unknown word
        1
    """
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token: int) -> str:
    """Convert a token ID back to its word string.

    Reverse lookup from token ID to the original word string.

    Args:
        token: The token ID to look up. Must be in valid range.

    Returns:
        The word corresponding to the token ID.

    Raises:
        IndexError: If token is out of vocabulary range.

    Example:
        >>> look_up_token(0)
        '<PAD>'
        >>> look_up_token(42)
        'the'
    """
    return _idx_to_word[token]


# =============================================================================
# TOKENIZATION
# =============================================================================

def _tokenize(string: str) -> list[str]:
    """Tokenize a string into lowercase words.

    Simple whitespace-based tokenization with lowercasing.

    Args:
        string: Input text string.

    Returns:
        List of lowercase word tokens.

    Example:
        >>> _tokenize("Hello World")
        ['hello', 'world']
    """
    return [word.lower() for word in string.split(" ")]


# =============================================================================
# BATCH PREPARATION
# =============================================================================

def _prepare_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Prepare a batch of QA examples for model training.

    Converts raw text data into padded numpy arrays suitable for
    feeding into a seq2seq model. Creates document tokens, answer
    masks, and question tokens with proper padding.

    Args:
        batch: List of story entry dicts, each containing:
            - document_id (str): Unique document identifier
            - document_text (str): Raw document text
            - document_words (list[str]): Tokenized document words
            - answer_text (str): Answer text span
            - answer_indices (list[int]): Word indices of answer in document
            - question_text (str): Question text
            - question_words (list[str]): Tokenized question words

    Returns:
        Prepared batch dictionary with numpy arrays:
            - size (int): Number of examples in batch
            - document_ids (list[str]): Document identifiers
            - document_text (list[str]): Raw document texts
            - document_words (list[list[str]]): Tokenized documents
            - document_tokens (np.ndarray): Shape (batch_size, max_doc_len)
            - document_lengths (np.ndarray): Shape (batch_size,)
            - answer_text (list[str]): Answer strings
            - answer_indices (list[list[int]]): Answer word indices
            - answer_labels (np.ndarray): Shape (batch_size, max_doc_len)
            - answer_masks (np.ndarray): Shape (batch_size, max_ans_len, max_doc_len)
            - answer_lengths (np.ndarray): Shape (batch_size,)
            - question_text (list[str]): Question strings
            - question_input_tokens (np.ndarray): Shape (batch_size, max_q_len)
            - question_output_tokens (np.ndarray): Shape (batch_size, max_q_len)
            - question_lengths (np.ndarray): Shape (batch_size,)
    """
    # Initialize collection lists
    id_to_indices: dict[str, list[int]] = {}
    document_ids: list[str] = []
    document_text: list[str] = []
    document_words: list[list[str]] = []
    answer_text: list[str] = []
    answer_indices: list[list[int]] = []
    question_text: list[str] = []
    question_input_words: list[list[str]] = []
    question_output_words: list[list[str]] = []

    # Extract and organize batch entries
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])

        # Add START/END tokens to question sequences
        q_words = entry["question_words"]
        question_input_words.append([START_WORD] + q_words)
        question_output_words.append(q_words + [END_WORD])

    # Calculate maximum sequence lengths for padding
    batch_size = len(batch)
    max_document_len = max(
        (len(doc) for doc in document_words),
        default=0
    )
    max_answer_len = max(
        (len(ans) for ans in answer_indices),
        default=0
    )
    max_question_len = max(
        (len(q) for q in question_input_words),
        default=0
    )

    # Initialize padded numpy arrays
    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros(
        (batch_size, max_answer_len, max_document_len),
        dtype=np.int32
    )
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros(
        (batch_size, max_question_len),
        dtype=np.int32
    )
    question_output_tokens = np.zeros(
        (batch_size, max_question_len),
        dtype=np.int32
    )
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    # Populate arrays with tokenized data
    for i in range(batch_size):
        # Tokenize document words
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        # Create answer labels and masks
        # Answer labels are shared across all questions for same document
        for j, idx in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, idx] = 1
            answer_masks[i, j, idx] = 1
        answer_lengths[i] = len(answer_indices[i])

        # Tokenize question sequences
        for j, word in enumerate(question_input_words[i]):
            question_input_tokens[i, j] = look_up_word(word)
        for j, word in enumerate(question_output_words[i]):
            question_output_tokens[i, j] = look_up_word(word)
        question_lengths[i] = len(question_input_words[i])

    return {
        "size": batch_size,
        "document_ids": document_ids,
        "document_text": document_text,
        "document_words": document_words,
        "document_tokens": document_tokens,
        "document_lengths": document_lengths,
        "answer_text": answer_text,
        "answer_indices": answer_indices,
        "answer_labels": answer_labels,
        "answer_masks": answer_masks,
        "answer_lengths": answer_lengths,
        "question_text": question_text,
        "question_input_tokens": question_input_tokens,
        "question_output_tokens": question_output_tokens,
        "question_lengths": question_lengths,
    }


# =============================================================================
# BATCH MANIPULATION UTILITIES
# =============================================================================

def collapse_documents(batch: dict[str, Any]) -> dict[str, Any]:
    """Remove duplicate documents from a batch.

    Keeps only the first occurrence of each unique document_id,
    useful for inference when you want one prediction per document
    rather than one per question.

    Args:
        batch: Prepared batch dict from _prepare_batch().

    Returns:
        Collapsed batch with unique documents only. All arrays are
        filtered to include only the kept indices.

    Example:
        >>> batch = _prepare_batch(stories)
        >>> batch['size']
        128
        >>> collapsed = collapse_documents(batch)
        >>> collapsed['size']
        45  # Unique documents only
    """
    seen_ids: set[str] = set()
    keep: list[int] = []

    # Identify first occurrence of each document
    for i in range(batch["size"]):
        doc_id = batch["document_ids"][i]
        if doc_id in seen_ids:
            continue
        keep.append(i)
        seen_ids.add(doc_id)

    # Filter all batch entries to kept indices
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "size":
            result[key] = len(keep)
        elif isinstance(value, np.ndarray):
            result[key] = value[keep]
        else:
            result[key] = [value[i] for i in keep]

    return result


def expand_answers(
    batch: dict[str, Any],
    answers: np.ndarray
) -> dict[str, Any]:
    """Expand predicted answer tags into answer spans.

    Converts binary answer tags (per-word predictions) into contiguous
    answer spans suitable for question generation. Takes the first
    contiguous span if multiple are predicted.

    Args:
        batch: Collapsed batch from collapse_documents().
        answers: Binary answer predictions of shape (batch_size, doc_len).
            Each position is 1 if that word is part of an answer.

    Returns:
        New prepared batch with expanded answer indices. Documents with
        no predicted answers get empty answer_indices.

    Example:
        >>> predictions = model.predict(batch)
        >>> answer_tags = (predictions > 0.5).astype(int)
        >>> expanded = expand_answers(batch, answer_tags)
    """
    new_batch: list[dict[str, Any]] = []

    for i in range(batch["size"]):
        # Find contiguous answer spans from binary predictions
        split_answers: list[list[int]] = []
        last: Optional[int] = None

        for j, tag in enumerate(answers[i]):
            if tag:
                # Start new span if not contiguous with previous
                if last != j - 1:
                    split_answers.append([])
                split_answers[-1].append(j)
                last = j

        # Create entry using first span (if any)
        if split_answers:
            ans_indices = split_answers[0]
            ans_text = " ".join(
                batch["document_words"][i][idx] for idx in ans_indices
            )
            new_batch.append({
                "document_id": batch["document_ids"][i],
                "document_text": batch["document_text"][i],
                "document_words": batch["document_words"][i],
                "answer_text": ans_text,
                "answer_indices": ans_indices,
                "question_text": "",
                "question_words": [],
            })
        else:
            # No answer predicted - use empty values
            new_batch.append({
                "document_id": batch["document_ids"][i],
                "document_text": batch["document_text"][i],
                "document_words": batch["document_words"][i],
                "answer_text": "",
                "answer_indices": [],
                "question_text": "",
                "question_words": [],
            })

    return _prepare_batch(new_batch)


# =============================================================================
# DATA LOADING
# =============================================================================

def _read_data(path: str) -> dict[str, list[dict[str, Any]]]:
    """Read QA data from a CSV file.

    Parses the CSV and groups entries by document_id for efficient
    batching of multiple questions about the same document.

    Args:
        path: Path to CSV file with columns:
            - document_id: Unique document identifier
            - document_text: Raw document text
            - question_text: Question about the document
            - answer_indices: Answer span as "start:end" format

    Returns:
        Dictionary mapping document_id to list of story entries.
        Each entry contains tokenized document, question, and answer.

    Note:
        Documents are truncated to _MAX_DOC_LENGTH words.
        Stories with no valid answer indices are skipped.
    """
    stories: dict[str, list[dict[str, Any]]] = {}

    with open(path, encoding="utf-8") as f:
        header_seen = False

        for row in csv.reader(f):
            # Skip header row
            if not header_seen:
                header_seen = True
                continue

            document_id = row[0]
            existing_stories = stories.setdefault(document_id, [])

            # Reuse tokenized document if same as previous entries
            document_text = row[1]
            if (existing_stories and
                    document_text == existing_stories[0]["document_text"]):
                # Share document data to save memory
                document_text = existing_stories[0]["document_text"]
                document_words = existing_stories[0]["document_words"]
            else:
                document_words = _tokenize(document_text)
                document_words = document_words[:_MAX_DOC_LENGTH]

            # Parse question
            question_text = row[2]
            question_words = _tokenize(question_text)

            # Parse answer span indices (format: "start:end,start:end,...")
            answer_str = row[3]
            answer_indices: list[int] = []
            for chunk in answer_str.split(","):
                start, end = (int(idx) for idx in chunk.split(":"))
                # Only include indices within document length limit
                if end < _MAX_DOC_LENGTH:
                    answer_indices.extend(range(start, end))

            # Extract answer text from document
            answer_text = " ".join(
                document_words[idx] for idx in answer_indices
            )

            # Skip entries with no valid answer
            if answer_indices:
                existing_stories.append({
                    "document_id": document_id,
                    "document_text": document_text,
                    "document_words": document_words,
                    "answer_text": answer_text,
                    "answer_indices": answer_indices,
                    "question_text": question_text,
                    "question_words": question_words,
                })

    return stories


# =============================================================================
# BATCH GENERATION
# =============================================================================

def _process_stories(
    stories: dict[str, list[dict[str, Any]]],
    batch_size: Optional[int] = None
) -> Generator[dict[str, Any], None, None]:
    """Generate batches from story data.

    Yields prepared batches by grouping stories up to batch_size.
    Stories are shuffled before batching for randomization.

    Args:
        stories: Dictionary mapping document_id to list of story entries.
        batch_size: Maximum examples per batch. Defaults to DEFAULT_BATCH_SIZE.

    Yields:
        Prepared batch dictionaries from _prepare_batch().

    Note:
        Stories from the same document are kept together in a batch
        to enable shared answer labeling.
    """
    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE

    batch: list[dict[str, Any]] = []
    story_groups = list(stories.values())
    random.shuffle(story_groups)

    for story_group in story_groups:
        # Yield current batch if adding this group would exceed limit
        if len(batch) + len(story_group) > batch_size:
            yield _prepare_batch(batch)
            batch = []
        batch.extend(story_group)

    # Yield final partial batch
    if batch:
        yield _prepare_batch(batch)


# =============================================================================
# DATA CACHING
# =============================================================================

# Cached story data (loaded on first access)
_training_stories: Optional[dict[str, list[dict[str, Any]]]] = None
_test_stories: Optional[dict[str, list[dict[str, Any]]]] = None


def _load_training_stories() -> dict[str, list[dict[str, Any]]]:
    """Load and cache training data stories.

    Loads stories from the training CSV file on first call,
    then returns cached data on subsequent calls.

    Returns:
        Dictionary mapping document_id to list of story entries.
    """
    global _training_stories
    train_path = str(_module_dir / "../../data/qa/train.csv")
    _training_stories = _read_data(train_path)
    return _training_stories


def _load_test_stories() -> dict[str, list[dict[str, Any]]]:
    """Load and cache test data stories.

    Loads stories from the test CSV file on first call,
    then returns cached data on subsequent calls.

    Returns:
        Dictionary mapping document_id to list of story entries.
    """
    global _test_stories
    test_path = str(_module_dir / "../../data/qa_test/my_test.csv")
    _test_stories = _read_data(test_path)
    return _test_stories


# =============================================================================
# PUBLIC DATA GENERATORS
# =============================================================================

def training_data(
    batch_size: Optional[int] = None
) -> Generator[dict[str, Any], None, None]:
    """Generate training data batches.

    Yields prepared batches from the training CSV file. Each batch
    contains up to batch_size examples with proper padding.

    Args:
        batch_size: Maximum examples per batch. Defaults to DEFAULT_BATCH_SIZE
            (384 for 8GB VRAM). Use 128 for 4GB, 256 for 6GB.

    Yields:
        Prepared batch dictionaries containing:
            - document_tokens: (batch_size, max_doc_len) token IDs
            - answer_labels: (batch_size, max_doc_len) binary labels
            - question_input_tokens: (batch_size, max_q_len) decoder input
            - question_output_tokens: (batch_size, max_q_len) decoder target
            - size: Number of examples in this batch
            - ... and other metadata (see _prepare_batch)

    Example:
        >>> for batch in training_data(batch_size=256):
        ...     loss = model.train_on_batch(
        ...         [batch['document_tokens'], batch['question_input_tokens']],
        ...         [batch['answer_labels'], batch['question_output_tokens']]
        ...     )
    """
    return _process_stories(_load_training_stories(), batch_size=batch_size)


def test_data(
    batch_size: Optional[int] = None
) -> Generator[dict[str, Any], None, None]:
    """Generate test data batches.

    Yields prepared batches from the test CSV file for evaluation.

    Args:
        batch_size: Maximum examples per batch. Defaults to DEFAULT_BATCH_SIZE.

    Yields:
        Prepared batch dictionaries (same format as training_data).

    Example:
        >>> for batch in test_data():
        ...     predictions = model.predict(batch['document_tokens'])
        ...     # Evaluate predictions...
    """
    return _process_stories(_load_test_stories(), batch_size=batch_size)


# =============================================================================
# EMBEDDING UTILITIES
# =============================================================================

def trim_embeddings() -> None:
    """Create trimmed GloVe embeddings file.

    Reads the full GloVe embeddings and writes a trimmed version
    containing only words that appear in the QA dataset vocabulary.
    This significantly reduces memory usage and load time.

    The trimmed vocabulary includes:
        - Top 5000 most frequent question words
        - Additional document words up to 10000 total

    Input file: data/glove/glove.6B.100d.txt (~862MB)
    Output file: data/glove/glove.6B.100d.trimmed.txt (~8MB)

    Note:
        Run this once after downloading the full GloVe embeddings.
        The trimmed file is required for module import.
    """
    # Count word frequencies in dataset
    document_counts: Counter[str] = Counter()
    question_counts: Counter[str] = Counter()

    for data in [_load_training_stories().values(),
                 _load_test_stories().values()]:
        for stories in data:
            if stories:
                document_counts.update(stories[0]["document_words"])
                for story in stories:
                    question_counts.update(story["question_words"])

    # Build vocabulary of words to keep
    keep: set[str] = set()

    # Include top 5000 question words
    for word, _ in question_counts.most_common(5000):
        keep.add(word)

    # Fill remaining slots with document words
    for word, _ in document_counts.most_common():
        if len(keep) >= 10000:
            break
        keep.add(word)

    # Write trimmed embeddings file
    glove_full_path = str(_module_dir / "../../data/glove/glove.6B.100d.txt")
    glove_trimmed_path = str(
        _module_dir / "../../data/glove/glove.6B.100d.trimmed.txt"
    )

    with open(glove_full_path, encoding="utf-8") as f_in:
        with open(glove_trimmed_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                word = line.split(" ")[0]
                if word in keep:
                    f_out.write(line)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # When run as script, create trimmed embeddings
    trim_embeddings()
