#!/usr/bin/env python
"""Question-Answer Data Processing Utilities.

This module provides utilities for loading, processing, and preparing
QA (Question-Answer) training data from CSV files. It includes:

- GloVe word embedding loading and vocabulary management
- Tokenization and batch preparation for seq2seq models
- Training and test data generators

The module loads trimmed GloVe embeddings at import time, which requires
the file `data/glove/glove.6B.100d.trimmed.txt` to exist. Run
`download_qa_data.sh` to create this file.

Typical usage::

    from src.utils.write import training_data, test_data, glove
    
    # Get pre-loaded GloVe embeddings
    print(f"Vocabulary size: {glove.shape[0]}")
    
    # Iterate over training batches
    for batch in training_data():
        document_tokens = batch['document_tokens']
        # ... process batch ...

Attributes:
    glove (np.ndarray): Pre-loaded GloVe embeddings matrix.
    PADDING_TOKEN (int): Token ID for padding.
    UNKNOWN_TOKEN (int): Token ID for unknown words.
    START_TOKEN (int): Token ID for sequence start.
    END_TOKEN (int): Token ID for sequence end.
"""

from collections import Counter
from pathlib import Path

import csv

import random

import numpy as np


_MAX_BATCH_SIZE = 128
_MAX_DOC_LENGTH = 200

PADDING_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
START_WORD = "<START>"
END_WORD = "<END>"

_word_to_idx = {}
_idx_to_word = []


def _add_word(word):
    """Add a word to the vocabulary and return its token ID.
    
    Args:
        word: The word string to add to vocabulary.
        
    Returns:
        int: The token ID assigned to the word.
    """
    idx = len(_idx_to_word)
    _word_to_idx[word] = idx
    _idx_to_word.append(word)
    return idx


PADDING_TOKEN = _add_word(PADDING_WORD)
UNKNOWN_TOKEN = _add_word(UNKNOWN_WORD)
START_TOKEN = _add_word(START_WORD)
END_TOKEN = _add_word(END_WORD)


# Calculate path relative to this module file (v1/src/utils/write.py)
# Data is in v1/data/, so we go up 2 levels from src/utils/ to v1/
_module_dir = Path(__file__).parent  # v1/src/utils/
embeddings_path = str(_module_dir / '../../data/glove/glove.6B.100d.trimmed.txt')

with open(embeddings_path) as f:
    line = f.readline()
    chunks = line.split(" ")
    dimensions = len(chunks) - 1
    f.seek(0)

    vocab_size = sum(1 for line in f)
    vocab_size += 4 #3 
    f.seek(0)

    glove = np.ndarray((vocab_size, dimensions), dtype=np.float32)
    glove[PADDING_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[UNKNOWN_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[START_TOKEN] = np.random.normal(0, 0.02, dimensions)
    glove[END_TOKEN] = np.random.normal(0, 0.02, dimensions)

    for line in f:
        chunks = line.split(" ")
        idx = _add_word(chunks[0])
        glove[idx] = [float(chunk) for chunk in chunks[1:]]
        if len(_idx_to_word) >= vocab_size:
            break




def look_up_word(word):
    """Convert a word to its token ID.
    
    Args:
        word: The word string to look up.
        
    Returns:
        int: Token ID for the word, or UNKNOWN_TOKEN if not in vocabulary.
    """
    return _word_to_idx.get(word, UNKNOWN_TOKEN)


def look_up_token(token):
    """Convert a token ID back to its word string.
    
    Args:
        token: The token ID to look up.
        
    Returns:
        str: The word corresponding to the token ID.
        
    Raises:
        IndexError: If token is out of vocabulary range.
    """
    return _idx_to_word[token]



def _tokenize(string):
    """Tokenize a string into lowercase words.
    
    Args:
        string: Input text string.
        
    Returns:
        list[str]: List of lowercase word tokens.
    """
    return [word.lower() for word in string.split(" ")]


def _prepare_batch(batch):
    """Prepare a batch of QA examples for model training.
    
    Converts raw text data into padded numpy arrays suitable for
    feeding into a seq2seq model. Creates document tokens, answer
    masks, and question tokens.
    
    Args:
        batch: List of dicts with keys: document_id, document_text,
            document_words, answer_text, answer_indices, question_text,
            question_words.
            
    Returns:
        dict: Prepared batch with numpy arrays:
            - document_tokens: (batch_size, max_doc_len) int32
            - answer_masks: (batch_size, max_ans_len, max_doc_len) int32
            - answer_labels: (batch_size, max_doc_len) int32
            - question_input_tokens: (batch_size, max_q_len) int32
            - question_output_tokens: (batch_size, max_q_len) int32
            - size: int, batch size
    """
    id_to_indices = {}
    document_ids = []
    document_text = []
    document_words = []
    answer_text = []
    answer_indices = []
    question_text = []
    question_input_words = []
    question_output_words = []
    for i, entry in enumerate(batch):
        id_to_indices.setdefault(entry["document_id"], []).append(i)
        document_ids.append(entry["document_id"])
        document_text.append(entry["document_text"])
        document_words.append(entry["document_words"])
        answer_text.append(entry["answer_text"])
        answer_indices.append(entry["answer_indices"])
        question_text.append(entry["question_text"])

        question_words = entry["question_words"]
        question_input_words.append([START_WORD] + question_words)
        question_output_words.append(question_words + [END_WORD])

    batch_size = len(batch)
    max_document_len = max((len(document) for document in document_words), default=0)
    max_answer_len = max((len(answer) for answer in answer_indices), default=0)
    max_question_len = max((len(question) for question in question_input_words), default=0)

    document_tokens = np.zeros((batch_size, max_document_len), dtype=np.int32)
    document_lengths = np.zeros(batch_size, dtype=np.int32)
    answer_labels = np.zeros((batch_size, max_document_len), dtype=np.int32)
    answer_masks = np.zeros((batch_size, max_answer_len, max_document_len), dtype=np.int32)
    answer_lengths = np.zeros(batch_size, dtype=np.int32)
    question_input_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_output_tokens = np.zeros((batch_size, max_question_len), dtype=np.int32)
    question_lengths = np.zeros(batch_size, dtype=np.int32)

    for i in range(batch_size):
        for j, word in enumerate(document_words[i]):
            document_tokens[i, j] = look_up_word(word)
        document_lengths[i] = len(document_words[i])

        for j, index in enumerate(answer_indices[i]):
            for shared_i in id_to_indices[batch[i]["document_id"]]:
                answer_labels[shared_i, index] = 1
            answer_masks[i, j, index] = 1
        answer_lengths[i] = len(answer_indices[i])

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


def collapse_documents(batch):
    """Remove duplicate documents from a batch.
    
    Keeps only the first occurrence of each unique document_id,
    useful for inference when you want one prediction per document.
    
    Args:
        batch: Prepared batch dict from _prepare_batch().
        
    Returns:
        dict: Collapsed batch with unique documents only.
    """
    seen_ids = set()
    keep = []

    for i in range(batch["size"]):
        id = batch["document_ids"][i]
        if id in seen_ids:
            continue

        keep.append(i)
        seen_ids.add(id)

    result = {}
    for key, value in batch.items():
        if key == "size":
            result[key] = len(keep)
        elif isinstance(value, np.ndarray):
            result[key] = value[keep]
        else:
            result[key] = [value[i] for i in keep]
    return result


def expand_answers(batch, answers):
    """Expand predicted answer tags into answer spans.
    
    Converts binary answer tags (per-word) into contiguous answer
    spans for question generation.
    
    Args:
        batch: Collapsed batch from collapse_documents().
        answers: (batch_size, doc_len) array of binary answer tags.
        
    Returns:
        dict: New prepared batch with expanded answer indices.
    """
    new_batch = []

    for i in range(batch["size"]):
        split_answers = []
        last = None
        for j, tag in enumerate(answers[i]):
            if tag:
                if last != j - 1:
                    split_answers.append([])
                split_answers[-1].append(j)
                last = j

        if len(split_answers) > 0:

            answer_indices = split_answers[0]
        # for answer_indices in split_answers:
            document_id = batch["document_ids"][i]
            document_text = batch["document_text"][i]
            document_words = batch["document_words"][i]
            answer_text = " ".join(document_words[i] for i in answer_indices)
            new_batch.append({
                "document_id": document_id,
                "document_text": document_text,
                "document_words": document_words,
                "answer_text": answer_text,
                "answer_indices": answer_indices,
                "question_text": "",
                "question_words": [],
            })
        else:
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


def _read_data(path):
    """Read QA data from a CSV file.
    
    Parses the CSV and groups entries by document_id for efficient
    batching of questions about the same document.
    
    Args:
        path: Path to CSV file with columns: document_id, document_text,
            question_text, answer_indices.
            
    Returns:
        dict: Mapping of document_id to list of story entries.
    """
    stories = {}

    with open(path) as f:
        header_seen = False
        for row in csv.reader(f):
            if not header_seen:
                header_seen = True
                continue

            document_id = row[0]

            existing_stories = stories.setdefault(document_id, [])

            document_text = row[1]
            if existing_stories and document_text == existing_stories[0]["document_text"]:
                # Save memory by sharing identical documents
                document_text = existing_stories[0]["document_text"]
                document_words = existing_stories[0]["document_words"]
            else:
                document_words = _tokenize(document_text)
                document_words = document_words[:_MAX_DOC_LENGTH]

            question_text = row[2]
            question_words = _tokenize(question_text)

            answer = row[3]
            answer_indices = []
            for chunk in answer.split(","):
                start, end = (int(index) for index in chunk.split(":"))
                if end < _MAX_DOC_LENGTH:
                    answer_indices.extend(range(start, end))
            answer_text = " ".join(document_words[i] for i in answer_indices)

            if len(answer_indices) > 0:
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


def _process_stories(stories):
    """Generate batches from story data.
    
    Yields prepared batches by grouping stories up to _MAX_BATCH_SIZE.
    Stories are shuffled before batching.
    
    Args:
        stories: Dict mapping document_id to list of story entries.
        
    Yields:
        dict: Prepared batch from _prepare_batch().
    """
    batch = []
    vals = list(stories.values())
    random.shuffle(vals)

    for story in vals:
        if len(batch) + len(story) > _MAX_BATCH_SIZE:
            yield _prepare_batch(batch)
            batch = []
        batch.extend(story)

    if batch:
        yield _prepare_batch(batch)


_training_stories = None
_test_stories = None

def _load_training_stories():
    """Load and cache training data stories."""
    global _training_stories
    train_path = str(_module_dir / "../../data/qa/train.csv")
    _training_stories = _read_data(train_path)
    return _training_stories


def _load_test_stories():
    """Load and cache test data stories."""
    global _test_stories
    test_path = str(_module_dir / "../../data/qa_test/my_test.csv")
    _test_stories = _read_data(test_path)
    return _test_stories


def training_data():
    """Generate training data batches.
    
    Yields prepared batches from the training CSV file.
    Each batch contains up to _MAX_BATCH_SIZE examples.
    
    Yields:
        dict: Prepared batch with document_tokens, answer_masks, etc.
    """
    return _process_stories(_load_training_stories())


def test_data():
    """Generate test data batches.
    
    Yields prepared batches from the test CSV file.
    
    Yields:
        dict: Prepared batch with document_tokens, answer_masks, etc.
    """
    return _process_stories(_load_test_stories())


def trim_embeddings():
    """Create trimmed GloVe embeddings file.
    
    Reads the full GloVe embeddings and writes a trimmed version
    containing only words that appear in the QA dataset vocabulary.
    Keeps the top 5000 question words and up to 10000 total words.
    
    Output file: data/glove/glove.6B.100d.trimmed.txt
    """
    document_counts = Counter()
    question_counts = Counter()
    for data in [_load_training_stories().values(), _load_test_stories().values()]:
        
        for stories in data:

            if len(stories) > 0:
                document_counts.update(stories[0]["document_words"])
                for story in stories:
                    question_counts.update(story["question_words"])

    keep = set()
    for word, count in question_counts.most_common(5000):
        keep.add(word)
    for word, count in document_counts.most_common():
        if len(keep) >= 10000:
            break
        keep.add(word)

    glove_full_path = str(_module_dir / "../../data/glove/glove.6B.100d.txt")
    glove_trimmed_path = str(_module_dir / "../../data/glove/glove.6B.100d.trimmed.txt")
    
    with open(glove_full_path) as f:
        with open(glove_trimmed_path, "w") as f2:
            for line in f:
                if line.split(" ")[0] in keep:
                    f2.write(line)


if __name__ == '__main__':
    trim_embeddings()
