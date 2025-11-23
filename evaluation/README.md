# Phishing Detection Benchmark & Evaluation Tool (TestRig)

A robust evaluation framework for testing Small Language Models (SLMs) and LLMs on phishing email detection tasks. This tool aggregates multiple open-source datasets, normalizes them, and runs benchmarks against your local or remote LLM to measure **Accuracy** and **Repeatability**.

## 🚀 Features

- **Multi-Dataset Aggregation**: Automatically finds, loads, and unifies multiple CSV datasets (e.g., Nigerian Fraud, Enron, SpamAssassin) into a single standardized schema.
- **Balanced Sampling**: Ensures fair testing by creating 50/50 splits of Benign vs. Malicious emails from the aggregated pool.
- **Custom LLM Client**: Connects to any OpenAI-compatible API (specifically optimized for LMStudio/Local LLMs).
- **Production-Grade Prompting**: Uses the exact same system prompt and logic as the Chrome Extension to ensure test validity.
- **Two-Stage Evaluation**:
    1.  **Accuracy & Security Metrics**: Measures how well the model correctly identifies phishing vs. legitimate emails, calculating **Accuracy**, **Precision**, **Recall**, **F1-Score**, **False Positive Rate (FPR)**, and **False Negative Rate (FNR)**.
    2.  **Repeatability**: Measures the stability of the model's scoring and reasoning by querying the same email multiple times.

## 🛠 How It Works

### 1. Dataset Loading (`dataset_loader.py`)
The tool scans the project root for CSV files and normalizes them. It handles different schemas:
*   **Detailed**: `sender`, `receiver`, `date`, `subject`, `body`, `urls`, `label`
*   **Simple**: `subject`, `body`, `label`
*   **Combined**: `text_combined`, `label` (Parses Subject/Sender from body if possible)

**Quality Filters:**
*   Automatically excludes emails with body content < 20 characters to remove empty/gibberish rows.
*   Unifies all labels to `0` (Benign) and `1` (Malicious).

All data is unified into a single DataFrame.

### 2. Sampling & Balancing
To prevent class imbalance bias:
1.  The unified dataset is split into Benign and Malicious groups.
2.  A random sample of size `N/2` is drawn from each group (where `N` is your sample size).
3.  These samples are combined and shuffled to create the final test set.

### 3. Evaluation Process (`benchmark_all.py`)
The benchmark script runs two tests:

#### A. Accuracy Test
*   Iterates through the balanced sample.
*   Sends each email to the LLM (including Subject, Sender, Body Snippet, and Attachments).
*   Parses the JSON response (handling local model quirks like `<think>` tags).
*   Compares the model's Score (>50 = Malicious) against the Ground Truth.
*   **Calculates Advanced Metrics**:
    *   **Confusion Matrix**: TP, TN, FP, FN
    *   **FPR (False Positive Rate)**: Critical for UX (blocking legit emails).
    *   **FNR (False Negative Rate)**: Critical for Security (missing threats).
    *   **Latency**: Average time per inference.

#### B. Repeatability Test
*   Selects a small subset (1 Benign, 1 Malicious) from the sample.
*   Queries the LLM `X` times for the *same* email.
*   Calculates:
    *   **Score Variance**: How much the numerical score fluctuates.
    *   **Reason Consistency**: How often the exact same set of reasons is returned.

## 📦 Usage

### Prerequisites
*   Python 3.x
*   Dependencies: `pandas`, `requests`
*   A running LLM server (e.g., LMStudio) on `http://localhost:1234/v1` OR an OpenAI API Key.

### 🚀 Quick Start: Run Full Paper Benchmarks

To replicate the paper's methodology (Overall Performance, False Positive Stress Test, and Obvious Phish Test), use the provided shell script:

```bash
./def-acc-logiq/evaluation/run_paper_benchmarks.sh
```

This runs 3 sequential experiments and saves standardized JSON reports.

### 🧪 Multi-Model Benchmarking

To benchmark a list of models (e.g., Qwen, Gemma, Phi-4) sequentially with manual model switching:

```bash
./def-acc-logiq/evaluation/run_multi_model_benchmark.sh
```

This script will pause and prompt you to load each model before running the test.

### Manual Benchmarking

Run the benchmark script directly for custom experiments:

```bash
python3 def-acc-logiq/evaluation/benchmark_all.py \
  --model "google/gemma-3-4b" \
  --sample-size 50 \
  --repeat-runs 5 \
  --seed 42
```

### CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--api-key` | `""` | API Key for external providers (e.g. OpenAI). Adds `Authorization: Bearer <key>` header. |
| `--model` | `local-model` | Name of the model to track in reports (e.g., "google/gemma-3-4b") |
| `--sample-size` | `50` | Total number of emails to test (will be split 50/50 benign/malicious) |
| `--repeat-runs` | `5` | Number of times to re-test specific emails for stability check |
| `--api-url` | `http://localhost:1234/v1` | URL of your LLM API endpoint |
| `--temperature` | `0.1` | Sampling temperature (0.0-1.0). Higher values increase creativity/variance. |
| `--seed` | `None` | Random seed (int) for reproducible sampling. If set, the exact same emails will be selected. |
| `--data-dir` | Project Root | Directory to scan for CSV datasets |

## 📊 Output

The tool outputs a real-time console report showing:
*   Email Source & Metadata (Subject, Sender, Body Snippet)
*   Model Prediction & Reasoning
*   Latency

At the end, it displays a **Confusion Matrix** and key metrics (Accuracy, Precision, Recall, F1, FPR, FNR) directly in the terminal.

It also saves a detailed JSON file (`benchmark_results_seed<SEED>_<timestamp>.json`) containing:
*   Configuration used
*   Per-sample detailed results
*   Repeatability metrics (Variance, Consistency %)

## 🧩 Project Structure

*   `benchmark_all.py`: Main entry point and orchestrator.
*   `dataset_loader.py`: Handles CSV parsing, schema mapping, and data cleaning.
*   `llm_client.py`: Manages API connections, prompting, and response parsing.
*   `config.py`: Shared constants (legacy support).
