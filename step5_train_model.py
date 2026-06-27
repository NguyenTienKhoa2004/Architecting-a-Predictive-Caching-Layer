import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from step3_3_dataset import TrafficSlidingWindowDataset

SEQ_LEN = 12
PRED_LEN = 3
FEATURES = 2000
BATCH_SIZE = 64
HIDDEN_SIZE = 64       
NUM_LAYERS = 2        
DROPOUT = 0.2          
LEARNING_RATE = 0.001
EPOCHS = 50
PATIENCE = 10         
GRAD_CLIP = 2.0        

class TrafficLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, pred_len, dropout):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, pred_len * output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(-1, self.pred_len, self.output_size)
        return out

def get_dataloaders():
    print("[1/3] Preparing data (DataLoader without shuffling)...")
    train_df = pd.read_parquet("data/train_scaled.parquet")
    val_df = pd.read_parquet("data/val_scaled.parquet")
    test_df = pd.read_parquet("data/test_scaled.parquet")
    
    val_lookback = train_df.iloc[-SEQ_LEN:]
    val_df_merged = pd.concat([val_lookback, val_df])
    
    test_lookback = val_df.iloc[-SEQ_LEN:]
    test_df_merged = pd.concat([test_lookback, test_df])
    
    train_dataset = TrafficSlidingWindowDataset(train_df.values, SEQ_LEN, PRED_LEN)
    val_dataset = TrafficSlidingWindowDataset(val_df_merged.values, SEQ_LEN, PRED_LEN)
    test_dataset = TrafficSlidingWindowDataset(test_df_merged.values, SEQ_LEN, PRED_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders()
    
    print("[2/3] Initializing LSTM model...")
    model = TrafficLSTM(
        input_size=FEATURES, 
        hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, 
        output_size=FEATURES, 
        pred_len=PRED_LEN, 
        dropout=DROPOUT
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n================ STARTING TRAINING ================")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
                mae = torch.mean(torch.abs(outputs - y_batch))
                val_mae += mae.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_mae /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss (MSE): {train_loss:.5f} | Val Loss (MSE): {val_loss:.5f} | Val MAE: {val_mae:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "data/best_lstm_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"🔥 Early Stopping triggered at Epoch {epoch+1}! Stopping training to prevent overfitting.")
                break
                
    print("\n[3/3] EVALUATING MODEL ON TEST SET (FINAL ACCEPTANCE)")
    model.load_state_dict(torch.load("data/best_lstm_model.pth"))
    model.eval()
    
    test_loss = 0.0
    test_mae = 0.0
    total_latency = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            start_time = time.perf_counter()
            outputs = model(x_batch)
            end_time = time.perf_counter()
            
            total_latency += (end_time - start_time)
            total_samples += x_batch.size(0)
            
            loss = criterion(outputs, y_batch)
            test_loss += loss.item() * x_batch.size(0)
            
            mae = torch.mean(torch.abs(outputs - y_batch))
            test_mae += mae.item() * x_batch.size(0)
            
    test_mse = test_loss / len(test_loader.dataset)
    test_rmse = np.sqrt(test_mse)
    test_mae_final = test_mae / len(test_loader.dataset)
    
    avg_latency_ms = (total_latency / total_samples) * 1000
    
    print("\n================ FINAL TEST SET EVALUATION ================")
    print(f"1. RMSE (Root Mean Square Error): {test_rmse:.5f}")
    print(f"2. MAE (Mean Absolute Error):     {test_mae_final:.5f}")
    print(f"3. Inference Latency:             {avg_latency_ms:.3f} ms / sample")
    print("===========================================================")
    print("Note: The above errors are on the normalized scale (0-1). To get actual requests, use the inverse_transform function!")

if __name__ == "__main__":
    main()
