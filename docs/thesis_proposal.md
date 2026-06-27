# Thesis Proposal

**Title:** Architecting a Predictive Caching Layer: Applied Time Series Forecasting for a Single-Node System

## 1. Introduction and Problem Statement
In the modern context of microservices and distributed systems, retrieving data from a data source becomes the major performance bottleneck. Existing caching solutions are completely reactive and somewhat passive, that means the application will load the requested data only in case of cache misses. When the number of requests suddenly increases due to the spike in traffic, it causes severe performance problems such as latency and computational overhead.

## 2. Research Objectives
This thesis focuses on applied AI and systems engineering to solve caching bottlenecks. The primary objectives are to:

* **Architect a Log Simulator:** Build an offline telemetry simulation module capable of parsing server request logs as sequential time series, acting as a Proof of Concept (POC) alternative to complex real-time ingestion.
* **Integrate Sequence Modeling:** Apply deep learning forecasting models to recognize patterns in historical access logs and predict upcoming data requests.
* **Optimize Node Performance:** Develop a proactive "pre-fetching" mechanism that loads predicted data into a standalone cache ahead of user requests, aiming to maximize Cache Hit Rates and minimize latency on a single node.

## 3. Scope of Study
The research scope is firmly grounded in software architecture and applied artificial intelligence.

* **Domain:** Time-series server telemetry and a single-node caching environment (e.g., a standalone Redis instance).
* **Boundaries:** The project focuses on the engineering, integration, and performance measurement of the caching pipeline. It explicitly excludes the theoretical derivation or mathematical proofing of novel AI algorithms, as well as the complexities of distributed cluster synchronization.
* **Dimensionality:** To bypass the curse of dimensionality and prevent Out-of-Memory (OOM) inference bottlenecks, the simulated key-space is strictly bounded to the top 2,000 most frequently accessed keys, effectively pruning the 99% long-tail request distribution.

## 4. Methodology and Technologies
* **Data Engineering Pipeline:** Raw production traces will be processed using on-the-fly streaming decompression (via the `zstandard` library) to maintain a minimal memory footprint. Parsed access events will be aggregated into fixed temporal step matrices and serialized into columnar Apache Parquet format to accelerate disk I/O during model ingestion.
* **AI Modeling:** Utilization of a sequence-based Long Short-Term Memory (LSTM) neural network, implemented via the PyTorch open-source framework. To rigorously prevent temporal data leakage, the dataset will undergo a strict chronological split (Train: 10 days, Validation: 2 days, Test: 2 days) entirely without random shuffling.
* **Infrastructure Deployment:** The PoC testbench, including the simulation of a single application server connected to a local Redis cache and the Al inference engine, will be developed and hosted on a Windows environment (leveraging WSL2 and Docker for containerized node simulation) to ensure stable system resource allocation and performance monitoring.
* **Evaluation Strategy:** Comparative stress-testing will be conducted. The proposed AI-driven caching layer will be benchmarked against a baseline reactive cache, measuring critical system metrics: Cache Hit Rate, Request Latency, and CPU/Memory overhead on the single-node setup.

## 5. Evaluation and Validation
To rigorously validate the proposed predictive caching layer, the evaluation methodology is divided into two primary dimensions: AI Model Performance and System-Level Optimization.

### 5.1 AI Model Metrics
The forecasting accuracy of the time-series models will be assessed in isolation before integration. This ensures the validity of the generated data requests:

* **Regression Metrics:** Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) will be utilized to measure the deviation between predicted request volumes and actual access logs.
* **Inference Latency:** The computational time required by the Al engine to generate predictions, ensuring the model itself does not introduce unacceptable delays into the pipeline.

### 5.2 System-Level Metrics Benchmarking
The AI-driven pre-fetching mechanism will be benchmarked against standard reactive caching policies (e.g., Least Recently Used (LRU) and Least Frequently Used (LFU)). The testbench will measure the following critical metrics under simulated traffic bursts:

* **Cache Hit Rate (CHR):** The primary indicator of successful pre-fetching accuracy.
* **Tail Latency (P95 and P99):** To verify that the predictive layer effectively mitigates latency spikes during high-load scenarios.
* **System Overhead:** Analysis of CPU/Memory utilization during the pre-fetching and prediction process.

### 5.3 Dataset and Simulation Environment
The evaluation will leverage production-grade access traces from Twitter caching datasets (specifically a 7-day base trace from Twemcache clusters). To satisfy a rigorous 14-day simulation horizon without introducing multi-cluster concept drift, the baseline trace will be expanded using a self-stitching time-shift technique, projecting the temporal sequence into the subsequent week while injecting controlled Gaussian noise (±3% to 5%) to emulate realistic access variance while preserving natural diurnal patterns. Stress testing will be conducted using automated load generators to emulate sudden traffic spikes and validate system resilience.

## 6. Proposed Timeline
* **Week 1: Data Engineering & Twitter Data Preprocessing (System Foundation)**
    * **Core Task:** Transform raw Twitter access traces into a machine-learning-ready matrix.
    * **Execution Details:**
        * Utilize the `cluster18.sort.sample10.zst` Twitter trace file.
        * Implement an on-the-fly streaming decompression script using the `zstandard` library to prevent RAM overflow. Filter for "get" requests and hash the data into 5-minute time-steps.
        * Execute "Long-tail Pruning" according to Zipf's Law: Trim the 99% tail, retaining only the Top 2,000 most popular Keys to prevent VRAM overflow.
        * Apply Self-Stitching Time-Shift algorithm: Duplicate the first 7 days, shift them forward by 7 days, and inject Gaussian Noise (±5%) to create a complete 14-day dataset without distribution breakage.
    * **Deliverable:** A highly optimized `.parquet` binary file containing an exact matrix of 4,032 rows (14 days) x 2,000 columns.

* **Week 2: AI Model Development (LSTM Network & Sliding Window)**
    * **Core Task:** Build the AI prediction engine strictly preventing data leakage.
    * **Execution Details:**
        * Perform a Strict Chronological Split without random shuffling: 10 days for Train, 2 days for Validation, and 2 days for Test.
        * Implement a PyTorch Data Loader configured with a Sliding Window. Accurately handle boundary overlaps: When predicting the first batches of the Validation set, input batches must be drawn from the tail of the Train set to build momentum and avoid Lookahead Bias.
        * Train the isolated LSTM network and evaluate metrics: Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and crucially, Inference Latency.
    * **Deliverable:** A machine-learning verified LSTM model ready for deployment.

* **Week 3: Environment Configuration (Single-Node Infrastructure Setup)**
    * **Core Task:** Establish the simulated single-node server environment.
    * **Execution Details:**
        * Configure a server infrastructure testbench using Docker containers and Windows WSL2.
        * Initialize a standalone local Redis instance to serve as the Cache.
        * Ensure strong environment isolation so that system performance metrics (CPU and Memory overhead) are not distorted by other Windows processes.
    * **Deliverable:** A single-node server infrastructure ready for connection.

* **Week 4: Integration & Proactive Prefetching Logic**
    * **Core Task:** Connect the AI brain to the cache muscle.
    * **Execution Details:**
        * Develop a Log Simulator to "replay" request data streams into the system.
        * Implement the "Proactive Pre-fetching" logic: The server will call the LSTM model to predict the Top 2,000 Keys for the upcoming minutes, then automatically preload these Keys into the Redis database before actual user requests arrive.
    * **Deliverable:** A continuous data pipeline flowing from Log Simulator → AI Inference → Redis Cache.

* **Week 5: Pipeline Debugging & Optimization**
    * **Core Task:** Audit latency and compress computational costs to the lowest possible level.
    * **Execution Details:**
        * Conduct end-to-end integration testing of the entire pipeline.
        * Analyze bottlenecks. Ensure the AI Inference Latency is strictly shorter than traditional data retrieval times, preventing the AI from inadvertently slowing down the system.
        * Optimize source code to keep CPU and RAM consumption viable for a single-node setup.
    * **Deliverable:** A fully completed, stable Proof of Concept (PoC) without bugs.

* **Week 6: Benchmarking & Stress Testing**
    * **Core Task:** Measure the practical value of the proposed system against legacy technologies.
    * **Execution Details:**
        * Launch automated load generators to inject traffic bursts and spikes, simulating network outages.
        * Pit the AI-Driven Cache directly against two passive management policies: Least Recently Used (LRU) and Least Frequently Used (LFU).
        * Measure and export to CSV three key metrics: Cache Hit Rate, Tail Latency (P95 and P99), and System Overhead (CPU/Memory).
    * **Deliverable:** Raw benchmarking dataset and performance evaluation charts.

* **Week 7: Thesis Writing - Architecture & Methodology Analysis**
    * **Core Task:** Translate the research into academic writing.
    * **Execution Details:**
        * Write the Introduction and Literature Review chapters. Skillfully cite the Facebook Video paper, leveraging it to explain why this thesis avoids complex Markov algorithms in favor of a minimalist single-node design.
        * Write the Methodology chapter: Detail the "Long-tail Pruning" and "Strict Chronological Split" techniques to demonstrate profound data science rigor to the defense committee.
    * **Deliverable:** A completed draft of the technical process chapters.

* **Week 8: Thesis Writing - Evaluation Report & Defense Preparation**
    * **Core Task:** Conclude the research and prepare for the thesis defense (Extended from the initial 7-week proposal to guarantee defense readiness).
    * **Execution Details:**
        * Process the charts from Week 6 into an evaluation report, clearly proving that the AI pre-fetching mechanism optimizes Cache Hit Rate more effectively than LRU/LFU.
        * Reiterate the strict adherence to the project Scope: Emphasize that the thesis intentionally bypassed complex mathematical equations or distributed cluster synchronization algorithms to focus 100% on Applied Engineering.
        * Prepare the summary presentation slides.
    * **Deliverable:** The finalized Thesis Manuscript, ready for the final defense.