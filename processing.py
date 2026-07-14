# preprocessing.py
import re
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Dict

EMOTIONS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
IDX_TO_EMOTION = {idx: emotion for idx, emotion in enumerate(EMOTIONS)}

def clean_text(text: str) -> str:
    """Basic cleaning of student input text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9\s']", "", text)
    return text

def build_vocab(texts: List[str], max_vocab_size: int = 5000) -> Dict[str, int]:
    """Builds a word-to-index vocabulary dictionary with special tokens."""
    word_counts = {}
    for text in texts:
        cleaned = clean_text(text)
        for word in cleaned.split():
            word_counts[word] = word_counts.get(word, 0) + 1
            
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, _ in sorted_words[:max_vocab_size - 2]:
        vocab[word] = len(vocab)
    return vocab

def tokenize_and_pad(text: str, vocab: Dict[str, int], max_len: int = 50) -> List[int]:
    """Cleans, tokenizes, maps to vocab indices, and pads/truncates to max_len."""
    cleaned = clean_text(text)
    tokens = cleaned.split()
    ids = [vocab.get(word, vocab["<UNK>"]) for word in tokens]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def generate_synthetic_dataset(filepath: str = "dataset/emotions.csv"):
    """Generates a starter seed dataset if none exists so training scripts run immediately."""
    import os
    if os.path.exists(filepath):
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Synthetic samples capturing distinct keyword traces for demo training
    data = [
        # Confused
        ("I do not understand how this works at all.", "Confused"),
        ("Everything is super confusing and unclear to me.", "Confused"),
        ("I am completely lost in this recursion module.", "Confused"),
        ("What does this error even mean? It makes no sense.", "Confused"),
        # Frustrated
        ("This is impossible! I absolutely hate coding this.", "Frustrated"),
        ("I'm so tired of trying to solve this bug, it's annoying.", "Frustrated"),
        ("I feel like giving up entirely on this topic.", "Frustrated"),
        ("This tool is completely broken and stupid.", "Frustrated"),
        # Curious
        ("Why does the execution context behave this way?", "Curious"),
        ("I want to explore deeper into how compilers work.", "Curious"),
        ("How does a Transformer model scale up attention?", "Curious"),
        ("I'm super excited to learn how backpropagation works.", "Curious"),
        # Confident
        ("I solved this algorithm issue easily!", "Confident"),
        ("I fully understand the concepts taught today.", "Confident"),
        ("I have completed the entire machine learning task.", "Confident"),
        ("This topic is simple once you practice it.", "Confident"),
        # Bored
        ("This lesson is incredibly boring and slow.", "Bored"),
        ("I feel sleepy reading this repetitive code.", "Bored"),
        ("Not interested in studying this module at all.", "Bored"),
        ("I am just sitting here zoning out of class.", "Bored"),
    ] * 20  # Duplicate to create a lightweight, trainable set of 400 entries
    
    df = pd.DataFrame(data, columns=["text", "emotion"])
    df.to_csv(filepath, index=False)
    print(f"Created synthetic training dataset with {len(df)} rows at {filepath}")