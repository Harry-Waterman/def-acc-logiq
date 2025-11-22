"""
Example script showing how to use the evaluation module.

This demonstrates how to evaluate SLM model responses against the phishing email dataset.
"""

from evaluation import evaluate_model_response, evaluate_batch, evaluate_from_dataset, print_evaluation_report
import json

# Example 1: Evaluate a single model response
print("Example 1: Single Response Evaluation")
print("-" * 60)

model_output_1 = {
    "label": "not_malicious",
    "reasons": [
        "normal order update about an existing purchase",
        "no request for money, banking details, or credentials"
    ]
}

ground_truth_1 = "legitimate"  # or "not_malicious" depending on dataset labels

# Define label mapping to normalize different label formats
label_mapping = {
    "phishing": "malicious",
    "legitimate": "not_malicious",
    "not_malicious": "not_malicious",
    "malicious": "malicious"
}

is_correct, details = evaluate_model_response(
    model_output_1,
    ground_truth_1,
    label_mapping=label_mapping
)

print(f"Prediction: {details['predicted']}")
print(f"Ground Truth: {details['ground_truth']}")
print(f"Correct: {is_correct}")
print(f"Reasons: {details['reasons']}")
print()

# Example 2: Evaluate a batch of responses
print("Example 2: Batch Evaluation")
print("-" * 60)

model_outputs = [
    {
        "label": "not_malicious",
        "reasons": ["normal order update"]
    },
    {
        "label": "malicious",
        "reasons": ["suspicious link", "urgent action required"]
    },
    {
        "label": "not_malicious",
        "reasons": ["legitimate business communication"]
    }
]

ground_truth_labels = ["legitimate", "phishing", "legitimate"]

results = evaluate_batch(
    model_outputs,
    ground_truth_labels,
    label_mapping=label_mapping
)

print_evaluation_report(results, detailed=True)
print()

# Example 3: Evaluate from dataset file
print("Example 3: Evaluation from Dataset File")
print("-" * 60)
print("To use this, ensure you have downloaded the dataset from Kaggle.")
print("Example usage:")
print("""
# Assuming you have model_outputs from your SLM model
model_outputs = [...]  # List of model predictions

# Evaluate against the dataset
results = evaluate_from_dataset(
    model_outputs=model_outputs,
    dataset_path="../dataset/phishing_email_dataset.csv",  # or .json
    label_column="label",  # adjust based on actual column name
    label_mapping=label_mapping
)

print_evaluation_report(results, detailed=False)
""")

