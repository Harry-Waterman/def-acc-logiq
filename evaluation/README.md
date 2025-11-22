# Model Evaluation Module

This module provides functions to evaluate SLM (Small Language Model) responses for phishing email detection accuracy.

## Dataset

**You do NOT need to commit the dataset to the repository.** The dataset files are excluded via `.gitignore`.

### Downloading the Dataset

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset
2. Place the dataset file(s) in the `dataset/` directory (at the project root)
3. The dataset files will be ignored by git (they're in `.gitignore`)

### Dataset Format

The evaluation functions support:
- CSV files (`.csv`)
- JSON files (`.json`)

The dataset should contain a column with ground truth labels (typically named `label` or similar).

## Usage

### Basic Example

```python
from evaluation import evaluate_model_response, evaluate_batch, print_evaluation_report

# Single response evaluation
model_output = {
    "label": "not_malicious",
    "reasons": [
        "normal order update about an existing purchase",
        "no request for money, banking details, or credentials"
    ]
}

is_correct, details = evaluate_model_response(
    model_output,
    ground_truth_label="legitimate"
)
```

### Batch Evaluation

```python
model_outputs = [
    {"label": "not_malicious", "reasons": [...]},
    {"label": "malicious", "reasons": [...]},
    # ... more predictions
]

ground_truth_labels = ["legitimate", "phishing", "legitimate", ...]

results = evaluate_batch(model_outputs, ground_truth_labels)
print_evaluation_report(results)
```

### Evaluation from Dataset File

```python
from evaluation import evaluate_from_dataset, print_evaluation_report

# After downloading the dataset
results = evaluate_from_dataset(
    model_outputs=your_model_predictions,
    dataset_path="../dataset/phishing_email_dataset.csv",
    label_column="label"  # adjust based on actual column name
)

print_evaluation_report(results, detailed=True)
```

### Label Mapping

If your dataset uses different label names than your model outputs, use label mapping:

```python
label_mapping = {
    "phishing": "malicious",
    "legitimate": "not_malicious",
    "not_malicious": "not_malicious",
    "malicious": "malicious"
}

results = evaluate_batch(
    model_outputs,
    ground_truth_labels,
    label_mapping=label_mapping
)
```

## Functions

### `evaluate_model_response(model_output, ground_truth_label, label_mapping=None)`

Evaluates a single model response.

**Returns:** `(is_correct: bool, details: dict)`

### `evaluate_batch(model_outputs, ground_truth_labels, label_mapping=None)`

Evaluates a batch of model responses.

**Returns:** Dictionary with accuracy metrics and detailed results

### `evaluate_from_dataset(model_outputs, dataset_path, label_column="label", label_mapping=None)`

Evaluates model outputs against a dataset file.

**Returns:** Dictionary with accuracy metrics

### `print_evaluation_report(evaluation_results, detailed=False)`

Prints a formatted evaluation report with accuracy metrics.

## Output Format

The evaluation functions return metrics including:
- Overall accuracy
- Total samples
- Correct/incorrect predictions
- Per-class accuracy (for each label type)
- Detailed results for each sample

## Installation

```bash
pip install -r requirements.txt
```

