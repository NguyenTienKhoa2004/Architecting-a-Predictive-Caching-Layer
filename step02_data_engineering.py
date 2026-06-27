import io
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import zstandard as zstd

INPUT_ZST = "data/cluster18.sort.sample10.zst"
OUTPUT_PARQUET = "twitter_cluster18_14days_5min.parquet"
TIME_STEP = 300  
TOP_K = 2000  

def process_streaming():
    traffic_matrix = defaultdict(lambda: defaultdict(int))
    key_frequency = defaultdict(int)

    print("[1/4] Streaming Zstandard file and binning into 5-minute intervals...")
    start_time = time.time()

    with open(INPUT_ZST, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="ascii", errors="ignore")
            
            for line in text_stream:
                if ",get," not in line and ",gets," not in line:
                    continue

                row = line.split(",", 6)
                
                if len(row) < 6:
                    continue

                op = row[5]
                if op != "get" and op != "gets":
                    continue

                try:
                    timestamp = int(row[0])
                    raw_key = row[1]
                except ValueError:
                    continue

                bucket_id = timestamp // TIME_STEP

                traffic_matrix[bucket_id][raw_key] += 1
                key_frequency[raw_key] += 1

    elapsed = time.time() - start_time
    print(f"  -> Streaming completed in {elapsed:.2f} seconds! Found {len(key_frequency):,} unique keys.")
    return traffic_matrix, key_frequency

def main():
    traffic_matrix, key_frequency = process_streaming()

    print(f"[2/4] Filtering top {TOP_K} keys...")
    top_keys = [
        k for k, v in sorted(key_frequency.items(), key=lambda item: item[1], reverse=True)[:TOP_K]
    ]
    top_keys_set = set(top_keys)

    print("[3/4] Building original Week 1 matrix (2,016 buckets)...")
    total_buckets_week1 = 2016  

    df_week1 = pd.DataFrame(0, index=range(total_buckets_week1), columns=top_keys)

    for b_id in range(total_buckets_week1):
        if b_id in traffic_matrix:
            for key, count in traffic_matrix[b_id].items():
                if key in top_keys_set:
                    df_week1.at[b_id, key] = count

    del traffic_matrix
    del key_frequency

    print("[4/4] Applying Data Augmentation: Duplicating -> Injecting 5% noise -> Stitching to 14 days...")
    df_week2 = df_week1.copy()
    df_week2.index = df_week2.index + total_buckets_week1

    noise = np.random.uniform(0.95, 1.05, size=df_week2.shape)
    df_week2 = (df_week2 * noise).round().astype(int)

    df_14_days = pd.concat([df_week1, df_week2], axis=0)

    df_14_days.to_parquet(OUTPUT_PARQUET)

    print("\n" + "=" * 55)
    print("🎉 WEEK 1 DATA ENGINEERING COMPLETED!")
    print(f"Exported file: {OUTPUT_PARQUET}")
    print(f"Tensor matrix shape: {df_14_days.shape} (Buckets x Keys)")
    print(f"Time axis boundaries: Bucket {df_14_days.index[0]} -> Bucket {df_14_days.index[-1]}")
    print(f"Top 1 traffic key: '{top_keys[0]}'")
    print("=" * 55)

if __name__ == "__main__":
    main()