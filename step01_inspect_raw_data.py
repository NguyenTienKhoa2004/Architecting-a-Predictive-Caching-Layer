import zstandard as zstd

with open("data/cluster18.sort.sample10.zst", "rb") as f:
    dctx = zstd.ZstdDecompressor()
    with dctx.stream_reader(f) as reader:
        peek_bytes = reader.read(2000)
        print(peek_bytes.decode("ascii", errors="ignore"))