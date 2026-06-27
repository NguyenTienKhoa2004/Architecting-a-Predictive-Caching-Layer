import io
from collections import defaultdict
import zstandard as zstd

INPUT_ZST = "data/cluster18.sort.sample10.zst"
TOP_K = 2000

print("Scanning dataset to verify Zipf's Law coverage...")
key_frequency = defaultdict(int)
total_gets = 0

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
            raw_key = row[1]
            key_frequency[raw_key] += 1
            total_gets += 1

total_unique_keys = len(key_frequency)
sorted_keys = sorted(key_frequency.values(), reverse=True)

top_2000_gets = sum(sorted_keys[:TOP_K])
coverage = (top_2000_gets / total_gets) * 100

print(f"\n{'='*50}")
print(f"Total unique keys:     {total_unique_keys:,}")
print(f"Total GET requests:    {total_gets:,}")
print(f"Top {TOP_K} keys GET:    {top_2000_gets:,}")
print(f"Coverage:              {coverage:.2f}%")
print(f"{'='*50}")

print(f"\nTop 2000 / {total_unique_keys:,} keys = {TOP_K/total_unique_keys*100:.4f}% of all keys")
print(f"But they handle {coverage:.2f}% of all traffic")
