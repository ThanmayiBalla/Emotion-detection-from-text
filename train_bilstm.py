# train_bilstm.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from processing import (
    clean_text, build_vocab, tokenize_and_pad, 
    generate_synthetic_dataset, EMOTION_TO_IDX, EMOTIONS
)

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = [tokenize_and_pad(t, vocab, max_len) for t in texts]
        self.labels = [EMOTION_TO_IDX[l] for l in labels]
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        return (
            torch.tensor(self.texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate forward and backward final hidden states
        out = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        out = self.fc(self.dropout(out))
        return out

def run_bilstm_training():
    os.makedirs("models", exist_ok=True)
    dataset_path = "dataset/emotions.csv"
    if not os.path.exists(dataset_path):
        generate_synthetic_dataset(dataset_path)
        
    df = pd.read_csv(dataset_path)
    
    vocab = build_vocab(df["text"].tolist())
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
        
    # Split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    
    train_dataset = EmotionDataset(train_df["text"].tolist(), train_df["emotion"].tolist(), vocab)
    val_dataset = EmotionDataset(val_df["text"].tolist(), val_df["emotion"].tolist(), vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    model = BiLSTMClassifier(len(vocab), embed_dim=128, hidden_dim=64, num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    
    best_loss = float('inf')
    epochs = 5
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Eval
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == targets).sum().item()
                
        val_loss /= len(val_loader)
        acc = correct / len(val_dataset)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "models/bilstm_model.pth")
            
    print("BiLSTM training completed! Model successfully saved to models/bilstm_model.pth")

if __name__ == "__main__":
    run_bilstm_training()