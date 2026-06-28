# Architecting a Predictive Caching Layer

**Applied Time Series Forecasting for a Single-Node System**

> An LSTM-based predictive caching engine that learns server access patterns from production Twitter traces to proactively pre-fetch data, reducing cache misses and latency on a single-node setup.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [Phase 1: Data Engineering](#phase-1-data-engineering-week-1)
  - [Phase 2: AI Model Development](#phase-2-ai-model-development-week-2)
- [Model Results](#model-results)
- [Project Structure](#project-structure)
- [How to Reproduce](#how-to-reproduce)
- [Tech Stack](#tech-stack)
- [Upcoming Work](#upcoming-work)

---

## Project Overview

Traditional caching mechanisms (LRU, LFU) are **reactive** — they only respond after a cache miss has already occurred. Under sudden traffic spikes, this causes cascading latency and unnecessary database load.

This thesis proposes a **proactive** approach: using an LSTM neural network trained on historical server access logs to **predict which keys will be requested in the next 15 minutes**, then pre-loading them into a Redis cache *before* users actually request them.

### Key Research Questions

1. Can a lightweight LSTM model accurately forecast multi-dimensional server traffic patterns?
2. Is the model's inference speed fast enough (~sub-millisecond) to not become a bottleneck itself?
3. Does AI-driven pre-fetching meaningfully outperform traditional reactive caching policies?

---

## Dataset

| Property | Value |
|---|---|
| **Source** | Twitter Twemcache Production Traces (Cluster 18) |
| **Dataset Links** | [GitHub Repository](https://github.com/twitter/cache-trace) <br> [Cluster Data](https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source/) |
| **Raw Format** | Zstandard-compressed CSV (~8 GB) |
| **Base Duration** | 7 days of real server access logs |
| **Augmented Duration** | 14 days (Data Augmentation 7d -> 14d) |
| **Time Resolution** | 5-minute bins (buckets) |
| **Final Matrix Shape** | 4,032 rows × 2,000 columns |
| **Rows** | Each row = one 5-minute time bin |
| **Columns** | Top 2,000 most frequently accessed cache keys |
| **Cell Values** | Number of `GET` requests for that key in that time bin |

### Why Only 2,000 Keys?

Server access patterns follow **Zipf's Law**, where a small fraction of keys receives a disproportionately large share of traffic. Verified against the actual dataset:

| Metric | Value |
|--------|-------|
| Total unique keys in Cluster 18 | 1,132,548 |
| Total GET requests | 1,315,040,171 |
| Top 2,000 keys (0.18% of all keys) | 727,361,295 GETs (**55.31%** of total traffic) |

This means the top 0.18% of keys handle over half of all server traffic, confirming a strong Zipf-like concentration. The 2,000-key boundary is a deliberate **trade-off between traffic coverage and computational feasibility** — the LSTM's fully-connected output layer scales linearly with the number of keys, and exceeding this threshold risks Out-of-Memory errors during training on consumer hardware.

> **Acknowledged Limitation:** The remaining ~45% of traffic is distributed across 1.13 million long-tail keys. Increasing `TOP_K` to improve coverage is identified as a direction for future work.

### Data Augmentation Strategy

The original 7-day trace is insufficient for a rigorous Train/Val/Test split. To extend to 14 days without introducing distribution shift (concept drift), we apply **a data augmentation method using time-shifting combined with Gaussian noise insertion.**:

1. **Duplicate** the 7-day matrix into a second copy.
2. **Shift** the time index of the copy to follow immediately after the original.
3. **Inject** uniform random noise (×0.95 to ×1.05) to every cell in the copy.

This preserves the natural **diurnal patterns** (day/night traffic cycles) while preventing the model from memorizing exact values.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE OVERVIEW                       │
│                                                                 │
│  Raw Twitter Traces (.zst)                                      │
│         │                                                       │
│         ▼                                                       │
│  ┌───────────────┐  Streaming decompression + 5-min binning     │
│  │ step01_inspect│  + Long-tail pruning (Top 2000 keys)         │
│  │ step02_data_en│  + Data Augmentation (7d → 14d)              │
│  └──────┬────────┘                                              │
│         │  .parquet (4032 × 2000)                               │
│         ▼                                                       │
│  ┌───────────────┐  Strict Chronological Split                  │
│  │ step03_split_d│  Train: 10 days │ Val: 2 days │ Test: 2 days │
│  └──────┬────────┘                                              │
│         ▼                                                       │
│  ┌───────────────┐  MinMaxScaler fit on Train ONLY              │
│  │ step04_scale_f│  (Preventing Lookahead Bias)                 │
│  └──────┬────────┘                                              │
│         ▼                                                       │
│  ┌───────────────┐  Sliding Window Dataset + DataLoader         │
│  │ step05_sliding│  (shuffle=False, boundary overlap handling)  │
│  └──────┬────────┘                                              │
│         ▼                                                       │
│  ┌───────────────┐  2-Layer LSTM + Training Loop                │
│  │ step06_train_m│  Early Stopping + Dropout + Grad Clipping    │
│  └──────┬────────┘                                              │
│         │                                                       │
│         ▼                                                       │
│    best_lstm_model.pth  ←  Deployable AI Brain                  │
│    scaler.joblib        ←  For inverse_transform                │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │          UPCOMING: Weeks 3-6                             │   │
│  │  Docker + Redis Single-Node → AI Pre-fetching Engine     │   │
│  │  → Benchmark vs LRU/LFU → Stress Testing                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Walkthrough

### Phase 1: Data Engineering (Week 1)

#### Step 1 — Raw Data Inspection (`step01_inspect_raw_data.py`)

A minimal script to peek at the first 2,000 bytes of the compressed `.zst` file, confirming the CSV schema before building the full parser.

#### Step 2 — Streaming Aggregation & Augmentation (`step02_data_engineering.py`)

This is the core data engineering script. It processes an 8 GB compressed file **without ever loading it fully into RAM** by using `zstandard` streaming decompression.

**Processing Pipeline:**

| Stage | Action | Purpose |
|-------|--------|---------|
| **[1/4]** | Stream-decompress `.zst` file line-by-line | Prevent RAM overflow on large traces |
| **[2/4]** | Rank all keys by total GET frequency, keep Top 2,000 | Long-tail pruning per Zipf's Law |
| **[3/4]** | Build Week 1 matrix: 2,016 bins × 2,000 keys | Map raw logs into structured time-series |
| **[4/4]** | Duplicate → noise inject (±5%) → stitch into 14 days | Data Augmentation for sufficient Train/Val/Test split |

**Output:** `twitter_cluster18_14days_5min.parquet` — a 4,032 × 2,000 matrix.

---

### Phase 2: AI Model Development (Week 2)

#### Step 3.1 — Strict Chronological Split (`step03_chronological_split.py`)

The dataset is split **strictly by time order** (no random shuffling) to prevent temporal data leakage:

| Split | Duration | Rows | Purpose |
|-------|----------|------|---------|
| **Train** | Days 1–10 | 0 – 2,879 | Model learns patterns |
| **Validation** | Days 11–12 | 2,880 – 3,455 | Early Stopping & hyperparameter tuning |
| **Test** | Days 13–14 | 3,456 – 4,031 | Final unbiased evaluation |

#### Step 3.2 — Feature Scaling (`step04_feature_scaling.py`)

Applies `MinMaxScaler(0, 1)` to normalize all features. **Critical design decision:** the scaler is fit **exclusively on the Train set**, then applied (transform-only) to Val and Test. This prevents **Lookahead Bias** — the model never sees future statistical information during training.

The fitted scaler object is saved as `scaler.joblib` for later use in `inverse_transform` (converting normalized predictions back to real request counts).

#### Step 3.3 — Sliding Window Dataset (`step05_sliding_window_dataset.py`)

Implements a PyTorch `Dataset` with a sliding window approach:

- **Input window (X):** 12 consecutive bins (= 1 hour of history)
- **Prediction target (Y):** Next 3 bins (= 15 minutes into the future)

**Boundary Overlap Handling:** When creating the Validation DataLoader, the last 12 bins of the Train set are prepended to the Validation data. This ensures the very first prediction in the Validation set has a full 1-hour lookback — without this, the model would have no context for its first few predictions.

**`shuffle=False` is enforced on all three DataLoaders** (Train, Val, Test). For time-series data, shuffling destroys the macro-temporal order that the LSTM's cell state relies on to learn long-range dependencies like diurnal cycles.

#### Step 4 — LSTM Training & Evaluation (`step06_train_lstm_model.py`)

**Model Architecture:**

| Component | Configuration |
|-----------|---------------|
| **Network** | 2-layer LSTM → Linear |
| **Input** | `[batch, 12, 2000]` — 12 time bins × 2,000 key features |
| **Hidden Size** | 64 |
| **Output** | `[batch, 3, 2000]` — 3 future bins × 2,000 key predictions |
| **Loss Function** | MSE (Mean Squared Error) |
| **Optimizer** | Adam (lr=0.001) |

**Regularization (Anti-Overfitting):**

| Technique | Value | Purpose |
|-----------|-------|---------|
| **Dropout** | 0.2 | Randomly deactivate 20% of neurons between LSTM layers |
| **Early Stopping** | Patience=10 | Stop if Val Loss doesn't improve for 10 consecutive epochs |
| **Gradient Clipping** | max_norm=2.0 | Prevent exploding gradients in deep recurrent networks |

---

## Model Results

Evaluated on the held-out **Test set** (Days 13–14, data never seen during training):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 0.05541 | ~5.5% average error on the normalized (0–1) scale |
| **MAE** | 0.02325 | ~2.3% mean absolute deviation |
| **Inference Latency** | 0.324 ms/sample | Fast enough for real-time pre-fetching decisions |

> **Note:** These errors are on the normalized scale. To convert to actual request counts, apply `scaler.inverse_transform()` using the saved `scaler.joblib`.

The sub-millisecond inference latency confirms that the LSTM model will **not** become a performance bottleneck when integrated into the live caching pipeline.

---

## Project Structure

```
Thesis_v2/
├── data/                                # (git-ignored) All data and model artifacts
│   ├── cluster18.sort.sample10.zst      # Raw Twitter trace (~8 GB)
│   ├── train_data.parquet               # Raw Train split (10 days)
│   ├── val_data.parquet                 # Raw Validation split (2 days)
│   ├── test_data.parquet                # Raw Test split (2 days)
│   ├── train_scaled.parquet             # Normalized Train (0-1)
│   ├── val_scaled.parquet               # Normalized Validation (0-1)
│   ├── test_scaled.parquet              # Normalized Test (0-1)
│   ├── scaler.joblib                    # Fitted MinMaxScaler for inverse_transform
│   └── best_lstm_model.pth              # Best LSTM weights (lowest Val Loss)
│
├── docs/
│   └── thesis_proposal.md               # Full thesis proposal document
│
├── step01_inspect_raw_data.py           # Quick inspection of raw .zst data
├── step02_data_engineering.py           # Streaming parser + augmentation pipeline
├── step03_chronological_split.py        # Chronological Train/Val/Test split
├── step04_feature_scaling.py            # MinMaxScaler normalization
├── step05_sliding_window_dataset.py     # PyTorch Dataset & DataLoader setup
├── step06_train_lstm_model.py           # LSTM architecture, training & evaluation
├── verify_zipf.py                       # Verify top 2000 keys coverage vs actual
│
├── .gitignore                           # Excludes data/, *.parquet, *.zst, *.pth
└── README.md                            # This file
```

---

## How to Reproduce

### Prerequisites

```bash
pip install torch pandas numpy scikit-learn joblib zstandard
```

### Step-by-Step Execution

```bash
# 1. Place the raw trace in data/ directory
#    (cluster18.sort.sample10.zst must be downloaded separately)

# 2. Parse raw traces into 14-day augmented matrix
python step02_data_engineering.py

# 3. Split into Train (10d) / Val (2d) / Test (2d)
python step03_chronological_split.py

# 4. Normalize features with MinMaxScaler
python step04_feature_scaling.py

# 5. (Optional) Verify Dataset & DataLoader shapes
python step05_sliding_window_dataset.py

# 6. Train LSTM and evaluate on Test set
python step06_train_lstm_model.py
```

After training completes, `data/best_lstm_model.pth` and `data/scaler.joblib` are the two key artifacts needed for the deployment phase.

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.x |
| **Deep Learning** | PyTorch |
| **Data Processing** | Pandas, NumPy |
| **Feature Scaling** | scikit-learn (MinMaxScaler) |
| **Compression** | zstandard (streaming decompression) |
| **Data Format** | Apache Parquet (columnar, fast I/O) |
| **Model Serialization** | PyTorch state_dict (`.pth`) + joblib (`.joblib`) |

---

## Upcoming Work

| Week | Milestone | Description |
|------|-----------|-------------|
| **Week 3** | Infrastructure Setup | Docker + WSL2 single-node environment with standalone Redis |
| **Week 4** | Integration & Pre-fetching | Log Simulator → LSTM Inference → Redis auto-preload pipeline |
| **Week 5** | Debugging & Optimization | End-to-end latency audit, CPU/RAM optimization |
| **Week 6** | Benchmarking & Stress Testing | AI Cache vs LRU vs LFU under traffic spikes |
| **Week 7-8** | Thesis Writing & Defense | Methodology documentation, evaluation report, presentation |

---

## References (IEEE Format)

### Architecture Benchmarking (Literature Review — Chapter 2)

> **[1]** Authors (see IEEE page), "A Predictive Caching Strategy Tailored to a Real-World Dataset," *IEEE Xplore*, 2024. [Online]. Available: https://ieeexplore.ieee.org/document/10644108
>
> **Usage:** Benchmarked against this work's use of LSTM for optimizing Cache Hit Rate. While effective, their reliance on Markov-based models introduces excessive computational overhead — a limitation our single-node architecture is specifically designed to avoid.

### LSTM Regularization & Dropout Design (AI Design — Chapter 3)

> **[2]** W. Zaremba, I. Sutskever, and O. Vinyals, "Recurrent Neural Network Regularization," *arXiv preprint arXiv:1409.2329*, 2014. [Online]. Available: https://arxiv.org/pdf/1409.2329
>
> **Usage:** Justifies the decision to apply Dropout only on non-recurrent (vertical) connections between LSTM layers, preserving long-term memory flow through the cell state while preventing overfitting.

### Data Preprocessing & Anti-Leakage Principles (Design & Implementation — Chapters 3 & 4)

> **[3]** LDS Team, "Mastering LSTMs for Time Series: When Deep Learning Beats Statistics," *Let's Data Science*, Nov. 2025. [Online]. Available: https://letsdatascience.com/blog/mastering-lstms-for-time-series-when-deep-learning-beats-statistics
>
> **Usage:** Supports the principle that the scaler (MinMaxScaler) must be fit exclusively on the Train set to prevent data leakage. Also explains LSTM's internal gating mechanism for resolving the Vanishing Gradient problem.

### Data Leakage Definition & Prevention (Implementation — Chapter 4)

> **[4]** T. Mucci, "What is data leakage in machine learning?," *IBM Think*. [Online]. Available: https://www.ibm.com/think/topics/data-leakage-machine-learning
>
> **Usage:** Provides an industry-standard definition of data leakage from IBM, reinforcing the Strict Chronological Split (10-2-2 days, no shuffling) methodology used in this project to prevent train-test contamination.

### Data Augmentation & Noise Injection for Time Series (Implementation — Chapter 4)

> **[5]** Zilliz, "What is the role of noise injection in data augmentation?," *Milvus AI Quick Reference*, 2026. [Online]. Available: https://milvus.io/ai-quick-reference/what-is-the-role-of-noise-injection-in-data-augmentation

> **[6]** Zilliz, "How is data augmentation applied to time-series data?," *Milvus AI Quick Reference*, 2026. [Online]. Available: https://milvus.io/ai-quick-reference/how-is-data-augmentation-applied-to-timeseries-data
>
> **Usage:** Validates the 7-day to 14-day augmentation method. Confirms that Gaussian Noise Injection (±5%) is an established technique for generating synthetic time-series data that closely approximates natural variance while preventing overfitting.

### 14-Day Rolling Window Justification (Implementation — Chapter 4)

> **[7]** Digital Applied Team, "AI Content Personalization at Scale: Real-Time Guide," *Digital Applied Blog*, Mar. 2026. [Online]. Available: https://www.digitalapplied.com/blog/ai-content-personalization-scale-dynamic-real-time
>
> **Usage:** Validates the choice of a **14-day data window** for training the predictive model. The article establishes three key principles that align with this thesis: **(1)** user profile features must be recomputed over a **rolling 14-day window** to capture full weekly behavioral cycles (e.g., Monday traffic differs from Friday traffic); **(2)** data older than 2 weeks becomes "stale" and degrades prediction quality; **(3)** models should be **retrained every 7 days** to track behavioral drift. These principles directly support this project's 14-day augmented dataset and 10-2-2 day Train/Val/Test split.
>
> **Key Architectural Difference:** The article applies the 14-day window to a **live production** content personalization system at the web frontend layer, with continuous streaming data refreshed weekly. This thesis applies the same temporal principle to a **Proof-of-Concept offline** backend caching layer (standalone Redis), using a fixed 14-day dataset augmented from 7 days of Twitter traces via time-shifting and noise injection. The prediction target also differs: they predict *user content preferences*, while this thesis predicts *cache key access frequency*.

---

## License

This project is part of an undergraduate thesis. All rights reserved.
