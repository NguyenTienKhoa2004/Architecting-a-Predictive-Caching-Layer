import pandas as pd
import os

INPUT_PARQUET = "twitter_cluster18_14days_5min.parquet"
TRAIN_PARQUET = "data/train_data.parquet"
VAL_PARQUET = "data/val_data.parquet"
TEST_PARQUET = "data/test_data.parquet"

def main():
    print("[1/3] Loading data from Parquet...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Original data shape: {df.shape}")

    print("[2/3] Performing Strict Chronological Split...")
    train_df = df.iloc[0:2880]
    
    val_df = df.iloc[2880:3456]
    
    test_df = df.iloc[3456:4032]

    print(f"  -> Train set: {train_df.shape} (Row {train_df.index[0]} to {train_df.index[-1]})")
    print(f"  -> Validation set: {val_df.shape} (Row {val_df.index[0]} to {val_df.index[-1]})")
    print(f"  -> Test set: {test_df.shape} (Row {test_df.index[0]} to {test_df.index[-1]})")

    print("[3/3] Exporting datasets to Parquet files...")
    os.makedirs("data", exist_ok=True)
    
    train_df.to_parquet(TRAIN_PARQUET)
    val_df.to_parquet(VAL_PARQUET)
    test_df.to_parquet(TEST_PARQUET)
    
    print("Completed! Saved 3 files to data/ directory.")

if __name__ == "__main__":
    main()
