import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficSlidingWindowDataset(Dataset):
    def __init__(self, data, seq_len=12, pred_len=3):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def main():
    print("[1/3] Loading scaled datasets...")
    train_df = pd.read_parquet("data/train_scaled.parquet")
    val_df = pd.read_parquet("data/val_scaled.parquet")
    test_df = pd.read_parquet("data/test_scaled.parquet")
    
    SEQ_LEN = 12
    PRED_LEN = 3
    BATCH_SIZE = 64
    
    print(f"[2/3] Handling boundary overlaps (Lookback {SEQ_LEN} bins for Val and Test)...")
    val_lookback = train_df.iloc[-SEQ_LEN:]
    val_df_merged = pd.concat([val_lookback, val_df])
    
    test_lookback = val_df.iloc[-SEQ_LEN:]
    test_df_merged = pd.concat([test_lookback, test_df])
    
    print("[3/3] Initializing PyTorch Datasets and DataLoaders...")
    train_dataset = TrafficSlidingWindowDataset(train_df.values, SEQ_LEN, PRED_LEN)
    val_dataset = TrafficSlidingWindowDataset(val_df_merged.values, SEQ_LEN, PRED_LEN)
    test_dataset = TrafficSlidingWindowDataset(test_df_merged.values, SEQ_LEN, PRED_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\n[Results]")
    print(f"Total samples in Train Dataset: {len(train_dataset)}")
    print(f"Total samples in Val Dataset:   {len(val_dataset)}")
    print(f"Total samples in Test Dataset:  {len(test_dataset)}")
    
    x_batch, y_batch = next(iter(train_loader))
    print(f"\n[Shape of 1 Train Batch]")
    print(f"X (Input) shape:  {x_batch.shape} -> [batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, features={x_batch.shape[2]}]")
    print(f"Y (Output) shape: {y_batch.shape} -> [batch_size={BATCH_SIZE}, pred_len={PRED_LEN}, features={y_batch.shape[2]}]")

if __name__ == "__main__":
    main()
