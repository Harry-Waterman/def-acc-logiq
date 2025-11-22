# Evaluation Module

Evaluation tools for assessing SLM model performance on phishing email detection.

## Overview

This module provides:
- **Accuracy Evaluation**: Compare model predictions against ground truth labels
- **Repeatability Benchmark**: Assess model consistency across multiple runs

## Configuration

All constants (labels, reasons, dataset paths) are defined in `config.py`. Update these when integrating with your real model:

- `LABELS`: Standard label definitions
- `LABEL_MAPPING`: Maps dataset labels to standard labels
- `EXAMPLE_REASONS`: Example reasons for testing (replace with actual model output)
- `DATASET_CONFIG`: Dataset file paths and column names
- `EVALUATION_CONFIG`: Default test parameters

## Quick Start

```python
from evaluation import evaluate_model_response, evaluate_batch, print_evaluation_report
from evaluation.repeatability import assess_repeatability, print_repeatability_report

# Evaluate a single prediction
is_correct, details = evaluate_model_response(
    model_output={"label": "not_malicious", "reasons": [...]},
    ground_truth_label="legitimate"
)

# Evaluate batch
results = evaluate_batch(model_outputs, ground_truth_labels)
print_evaluation_report(results)

# Test repeatability
repeatability_result = assess_repeatability(
    your_model_function,
    "email text",
    num_runs=100
)
print_repeatability_report(repeatability_result)
```

## Dataset

**Location**: `dataset/Nigerian_Fraud.csv` (not committed to git)

**Format**: CSV with columns:
- `sender`
- `receiver`
- `date`
- `subject`
- `body`
- `urls`
- `label` (0 = legitimate, 1 = phishing)

**Source**: Nigerian Fraud email dataset (loaded locally in `dataset/Nigerian_Fraud.csv`)

## Accuracy Evaluation

### Functions

- `evaluate_model_response(model_output, ground_truth_label, label_mapping=None)` - Single prediction
- `evaluate_batch(model_outputs, ground_truth_labels, label_mapping=None)` - Batch evaluation
- `evaluate_from_dataset(model_outputs, dataset_path, label_column="label", label_mapping=None)` - From CSV/JSON
- `print_evaluation_report(results, detailed=False)` - Print formatted report

### Label Mapping

Normalize different label formats:

```python
label_mapping = {
    "phishing": "malicious",
    "legitimate": "not_malicious",
    "not_malicious": "not_malicious",
    "malicious": "malicious"
}
```

### Example

```python
from evaluation import evaluate_from_dataset, print_evaluation_report

results = evaluate_from_dataset(
    model_outputs=your_predictions,
    dataset_path="dataset/Nigerian_Fraud.csv",
    label_column="label",
    label_mapping={0: "not_malicious", 1: "malicious"}
)

print_evaluation_report(results, detailed=True)
```

## Repeatability Benchmark

Assesses how consistent model outputs are when given the same input multiple times.

### Functions

- `assess_repeatability(model_function, test_input, num_runs=100, label_mapping=None)` - Single input
- `benchmark_repeatability(model_function, test_inputs, num_runs=100, label_mapping=None)` - Multiple inputs
- `print_repeatability_report(result, detailed=False)` - Print report

### Metrics

- **Consistency Score**: 0-100% (percentage of most common prediction)
  - >95%: HIGHLY_REPEATABLE
  - 80-94%: REPEATABLE
  - 60-79%: MODERATELY_REPEATABLE
  - <60%: NOT_REPEATABLE
- **Agreement Rate**: Normalized consistency (0.0-1.0)
- **Normalized Entropy**: Uncertainty measure (lower is better)

### Example

```python
from evaluation.repeatability import assess_repeatability, print_repeatability_report

def your_model(input_text):
    # Your SLM model call
    return {"label": "malicious", "reasons": [...]}

result = assess_repeatability(
    your_model,
    "test email",
    num_runs=100
)

print_repeatability_report(result)
```

## Test Scripts

### Accuracy Evaluation
```bash
python evaluation/test_with_dataset.py
```
Tests accuracy evaluation with actual dataset emails.

### Repeatability Benchmark
```bash
python evaluation/test_repeatability_with_dataset.py
```
Tests repeatability with actual dataset emails.

## Node.js Benchmark App

The `evaluation/node-app` package provides an Express server and CLI to run the
LLM against the CSV datasets instead of live Outlook traffic.

### Features
- Load emails from any CSV that matches the configured schema
- Run the model N times per email to measure accuracy and repeatability
- REST endpoint (`POST /api/benchmark/run`) for automation
- CLI helper: `npm run benchmark -- --sampleSize 10 --numRuns 50`

### Setup
```bash
cd evaluation/node-app
npm install
cp env.example .env            # optional overrides
npm run dev                    # starts http://localhost:4100
```

You can either run the evaluator directly (requires you to start an `mlc-llm serve ...` process)
or use the bundled Docker Compose file (`evaluation/node-app/docker-compose.yml`) which launches an
mlc-llm container plus the evaluator in one command. See the node-app README for details.

By default the service calls an external [mlc-llm](https://github.com/mlc-ai/mlc-llm) REST server hosting
the same Qwen model + system prompt used by the Chrome extension. That keeps the evaluator in sync with
the real inference pipeline without needing a browser runtime.

## Integration

### Model Output Format

Your model should return:
```python
{
    "label": "malicious" | "not_malicious",
    "reasons": ["reason 1", "reason 2", ...]
}
```

### Wrapper Function

```python
def my_slm_model(input_text):
    response = your_slm_api_call(input_text)
    return {
        "label": response["label"],
        "reasons": response.get("reasons", [])
    }
```

## Installation

```bash
pip install -r requirements.txt
```

## Best Practices

1. **Use actual dataset**: Test with real emails from the dataset
2. **Sufficient runs**: Use at least 100 runs for repeatability tests
3. **Monitor metrics**: Track accuracy and consistency over time
4. **Set thresholds**: Define minimum acceptable performance (e.g., >95% consistency)
5. **Label normalization**: Use label mapping to handle different label formats
