# utils.py
import os
import csv
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt

LOGS_FILE = "logs/emotion_logs.csv"

def initialize_logs():
    """Sets up logs file structure with robust header attributes."""
    if not os.path.exists(LOGS_FILE):
        os.makedirs(os.path.dirname(LOGS_FILE), exist_ok=True)
        with open(LOGS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Student Text", "Primary Emotion", 
                "Secondary Emotion", "BiLSTM Prediction", "BERT Prediction", 
                "Confidence", "Gemini Response"
            ])

def log_interaction(text: str, prediction_data: Dict[str, Any], gemini_response: str):
    """Appends transactions to log file securely."""
    initialize_logs()
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bilstm_top = max(prediction_data["bilstm_probabilities"], key=prediction_data["bilstm_probabilities"].get)
    bert_top = max(prediction_data["bert_probabilities"], key=prediction_data["bert_probabilities"].get)
    
    with open(LOGS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            text.replace("\n", " "),
            prediction_data["primary_emotion"],
            prediction_data["secondary_emotion"],
            bilstm_top,
            bert_top,
            round(prediction_data["primary_confidence"], 4),
            gemini_response.replace("\n", " [NEWLINE] ")
        ])

def read_logs() -> pd.DataFrame:
    """Reads logs safely."""
    initialize_logs()
    try:
        df = pd.read_csv(LOGS_FILE)
        # Parse potential multi-line sequences safely
        if "Gemini Response" in df.columns:
            df["Gemini Response"] = df["Gemini Response"].str.replace(" [NEWLINE] ", "\n", regex=False)
        return df
    except Exception:
        return pd.DataFrame()

def clear_logs():
    """Empties logs file."""
    if os.path.exists(LOGS_FILE):
        os.remove(LOGS_FILE)
    initialize_logs()

def render_sample_loss_curves():
    """Generates a standard model training convergence plot dynamically."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    epochs = [1, 2, 3, 4, 5]
    train_loss = [0.85, 0.52, 0.31, 0.18, 0.11]
    val_loss = [0.72, 0.48, 0.35, 0.28, 0.26]
    
    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss')
    ax1.plot(epochs, val_loss, 'r--s', label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.legend()
    ax1.grid(True)
    
    train_acc = [0.65, 0.78, 0.88, 0.93, 0.96]
    val_acc = [0.68, 0.77, 0.84, 0.87, 0.89]
    
    ax2.plot(epochs, train_acc, 'b-o', label='Train Acc')
    ax2.plot(epochs, val_acc, 'r--s', label='Val Acc')
    ax2.set_title('Accuracy curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig