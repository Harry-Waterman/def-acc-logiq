"""
Test script for the evaluation module.
Tests the provided model output.
"""

import sys
from pathlib import Path

# Add the correct directory to path so we can import evaluation
# Handle both cases: running from root or from evaluation folder
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent

# Check if we're in def-acc-logiq/evaluation or def-acc-hackathon/def-acc-logiq/evaluation
if parent_dir.name == 'def-acc-logiq':
    # We're in def-acc-logiq/evaluation, add def-acc-logiq to path
    sys.path.insert(0, str(parent_dir))
elif parent_dir.parent.name == 'def-acc-logiq':
    # We're in def-acc-hackathon/def-acc-logiq/evaluation, add def-acc-logiq to path
    sys.path.insert(0, str(parent_dir))
else:
    # Fallback: add parent directory
    sys.path.insert(0, str(parent_dir))

# Import only functions that don't require pandas for basic testing
try:
    from evaluation import evaluate_model_response, evaluate_batch, print_evaluation_report
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")
    print("Testing basic functions only...")
    # For basic testing, we can test the core logic without pandas
    import json
    from typing import Dict, List, Tuple, Union
    
    def evaluate_model_response(
        model_output: Union[str, Dict],
        ground_truth_label: str,
        label_mapping: Dict[str, str] = None
    ) -> Tuple[bool, Dict]:
        """Simplified version for testing without pandas."""
        if isinstance(model_output, str):
            try:
                model_output = json.loads(model_output)
            except json.JSONDecodeError:
                return False, {"error": "Invalid JSON format"}
        
        predicted_label = model_output.get("label", "").lower().strip()
        gt_label = ground_truth_label.lower().strip()
        
        if label_mapping:
            predicted_label = label_mapping.get(predicted_label, predicted_label)
            gt_label = label_mapping.get(gt_label, gt_label)
        
        is_correct = predicted_label == gt_label
        
        details = {
            "predicted": predicted_label,
            "ground_truth": gt_label,
            "correct": is_correct,
            "reasons": model_output.get("reasons", [])
        }
        
        return is_correct, details
    
    def evaluate_batch(
        model_outputs: List[Union[str, Dict]],
        ground_truth_labels: List[str],
        label_mapping: Dict[str, str] = None
    ) -> Dict:
        """Simplified version for testing without pandas."""
        if len(model_outputs) != len(ground_truth_labels):
            raise ValueError("Number of model outputs must match number of ground truth labels")
        
        results = []
        correct_count = 0
        
        for model_output, gt_label in zip(model_outputs, ground_truth_labels):
            is_correct, details = evaluate_model_response(model_output, gt_label, label_mapping)
            results.append(details)
            if is_correct:
                correct_count += 1
        
        accuracy = correct_count / len(results) if results else 0.0
        
        class_stats = {}
        for result in results:
            gt = result["ground_truth"]
            if gt not in class_stats:
                class_stats[gt] = {"total": 0, "correct": 0}
            class_stats[gt]["total"] += 1
            if result["correct"]:
                class_stats[gt]["correct"] += 1
        
        per_class_accuracy = {
            label: stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            for label, stats in class_stats.items()
        }
        
        return {
            "accuracy": accuracy,
            "total_samples": len(results),
            "correct_predictions": correct_count,
            "incorrect_predictions": len(results) - correct_count,
            "per_class_accuracy": per_class_accuracy,
            "detailed_results": results
        }
    
    def print_evaluation_report(evaluation_results: Dict, detailed: bool = False):
        """Simplified version for testing without pandas."""
        print("=" * 60)
        print("MODEL EVALUATION REPORT")
        print("=" * 60)
        print(f"\nOverall Accuracy: {evaluation_results['accuracy']:.2%}")
        print(f"Total Samples: {evaluation_results['total_samples']}")
        print(f"Correct Predictions: {evaluation_results['correct_predictions']}")
        print(f"Incorrect Predictions: {evaluation_results['incorrect_predictions']}")
        
        if evaluation_results.get('per_class_accuracy'):
            print("\nPer-Class Accuracy:")
            for label, acc in evaluation_results['per_class_accuracy'].items():
                print(f"  {label}: {acc:.2%}")
        
        if detailed and evaluation_results.get('detailed_results'):
            print("\nDetailed Results:")
            print("-" * 60)
            for i, result in enumerate(evaluation_results['detailed_results'][:10]):
                status = "[PASS]" if result['correct'] else "[FAIL]"
                print(f"{status} Sample {i+1}: Predicted '{result['predicted']}' | "
                      f"Ground Truth '{result['ground_truth']}'")
            if len(evaluation_results['detailed_results']) > 10:
                print(f"... and {len(evaluation_results['detailed_results']) - 10} more samples")
        
        print("=" * 60)

# Test the provided model output
model_output = {
    "label": "not_malicious",
    "reasons": [
        "normal order update about an existing purchase",
        "no request for money, banking details, or credentials"
    ]
}

print("=" * 60)
print("TESTING EVALUATION MODULE")
print("=" * 60)

# Test Case 1: Correct prediction (not_malicious)
print("\nTest Case 1: Correct prediction (not_malicious)")
print("-" * 60)
is_correct, details = evaluate_model_response(
    model_output,
    ground_truth_label="not_malicious"
)
print(f"Predicted: {details['predicted']}")
print(f"Ground Truth: {details['ground_truth']}")
print(f"Correct: {is_correct} [PASS]" if is_correct else f"Correct: {is_correct} [FAIL]")
print(f"Reasons: {details['reasons']}")

# Test Case 2: Correct prediction with label mapping (legitimate -> not_malicious)
print("\nTest Case 2: Correct prediction with label mapping (legitimate -> not_malicious)")
print("-" * 60)
label_mapping = {
    "phishing": "malicious",
    "legitimate": "not_malicious",
    "not_malicious": "not_malicious",
    "malicious": "malicious"
}
is_correct, details = evaluate_model_response(
    model_output,
    ground_truth_label="legitimate",
    label_mapping=label_mapping
)
print(f"Predicted: {details['predicted']}")
print(f"Ground Truth: {details['ground_truth']}")
print(f"Correct: {is_correct} [PASS]" if is_correct else f"Correct: {is_correct} [FAIL]")
print(f"Reasons: {details['reasons']}")

# Test Case 3: Incorrect prediction (should be malicious but predicted not_malicious)
print("\nTest Case 3: Incorrect prediction (ground truth is malicious)")
print("-" * 60)
is_correct, details = evaluate_model_response(
    model_output,
    ground_truth_label="malicious",
    label_mapping=label_mapping
)
print(f"Predicted: {details['predicted']}")
print(f"Ground Truth: {details['ground_truth']}")
print(f"Correct: {is_correct} [PASS]" if is_correct else f"Correct: {is_correct} [FAIL]")
print(f"Reasons: {details['reasons']}")

# Test Case 4: Batch evaluation
print("\nTest Case 4: Batch evaluation")
print("-" * 60)
model_outputs = [
    model_output,  # The provided output
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

print("\n" + "=" * 60)
print("TESTING COMPLETE")
print("=" * 60)

