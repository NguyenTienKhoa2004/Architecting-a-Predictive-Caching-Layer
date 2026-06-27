import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler

TRAIN_PARQUET = "data/train_data.parquet"
VAL_PARQUET = "data/val_data.parquet"
TEST_PARQUET = "data/test_data.parquet"

TRAIN_SCALED = "data/train_scaled.parquet"
VAL_SCALED = "data/val_scaled.parquet"
TEST_SCALED = "data/test_scaled.parquet"

SCALER_FILE = "data/scaler.joblib"

def main():
    print("[1/4] Loading Train, Val, and Test datasets...")
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df = pd.read_parquet(VAL_PARQUET)
    test_df = pd.read_parquet(TEST_PARQUET)

    print(f"Train set shape: {train_df.shape}")
    print(f"Val set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")

    print("\n[2/4] Initializing and Fitting Scaler ONLY on Train set (Preventing Lookahead Bias)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaler.fit(train_df)

    print("[3/4] Transforming (scaling) data for all 3 datasets...")
    train_scaled = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns, index=train_df.index)
    val_scaled = pd.DataFrame(scaler.transform(val_df), columns=val_df.columns, index=val_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

    print("\n[4/4] Saving scaled data and Scaler model...")
    train_scaled.to_parquet(TRAIN_SCALED)
    val_scaled.to_parquet(VAL_SCALED)
    test_scaled.to_parquet(TEST_SCALED)
    
    joblib.dump(scaler, SCALER_FILE)

    print("Completed! Files have been saved to:")
    print(f" - {TRAIN_SCALED}")
    print(f" - {VAL_SCALED}")
    print(f" - {TEST_SCALED}")
    print(f" - {SCALER_FILE} (Use for inverse_transform later)")

if __name__ == "__main__":
    main()
