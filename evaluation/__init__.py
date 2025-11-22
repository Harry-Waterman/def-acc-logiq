"""
Evaluation module for SLM model responses on phishing email detection.

This module provides functions to evaluate model accuracy by comparing
model predictions against ground truth labels from the dataset.
"""

import json
from typing import Dict, List, Tuple, Union
from pathlib import Path
import pandas as pd


def evaluate_model_response(
    model_output: Union[str, Dict],
    ground_truth_label: str,
    label_mapping: Dict[str, str] = None
) -> Tuple[bool, Dict]:
    """
    Evaluate a single model response against ground truth.
    
    Args:
        model_output: Model output as JSON string or dict with 'label' and 'reasons' keys
        ground_truth_label: True label from dataset (e.g., 'phishing', 'legitimate', 'not_malicious')
        label_mapping: Optional mapping to normalize labels (e.g., {'phishing': 'malicious', 'legitimate': 'not_malicious'})
    
    Returns:
        Tuple of (is_correct: bool, details: dict)
    """
    # Parse model output if it's a string
    if isinstance(model_output, str):
        try:
            model_output = json.loads(model_output)
        except json.JSONDecodeError:
            return False, {"error": "Invalid JSON format"}
    
    # Extract predicted label
    predicted_label = model_output.get("label", "").lower().strip()
    
    # Normalize ground truth label
    gt_label = ground_truth_label.lower().strip()
    
    # Apply label mapping if provided
    if label_mapping:
        predicted_label = label_mapping.get(predicted_label, predicted_label)
        gt_label = label_mapping.get(gt_label, gt_label)
    
    # Check if prediction is correct
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
    """
    Evaluate a batch of model responses.
    
    Args:
        model_outputs: List of model outputs (JSON strings or dicts)
        ground_truth_labels: List of true labels
        label_mapping: Optional mapping to normalize labels
    
    Returns:
        Dictionary with accuracy metrics and detailed results
    """
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
    
    # Calculate per-class metrics
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


def evaluate_from_dataset(
    model_outputs: List[Union[str, Dict]],
    dataset_path: Union[str, Path],
    label_column: str = "label",
    label_mapping: Dict[str, str] = None
) -> Dict:
    """
    Evaluate model outputs against a dataset file (CSV, JSON, etc.).
    
    Args:
        model_outputs: List of model outputs
        dataset_path: Path to dataset file
        label_column: Name of the column containing ground truth labels
        label_mapping: Optional mapping to normalize labels
    
    Returns:
        Dictionary with accuracy metrics
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    # Load dataset based on file extension
    if dataset_path.suffix == '.csv':
        df = pd.read_csv(dataset_path)
    elif dataset_path.suffix == '.json':
        df = pd.read_json(dataset_path)
    else:
        raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
    
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataset")
    
    ground_truth_labels = df[label_column].tolist()
    
    return evaluate_batch(model_outputs, ground_truth_labels, label_mapping)


def print_evaluation_report(evaluation_results: Dict, detailed: bool = False):
    """
    Print a formatted evaluation report.
    
    Args:
        evaluation_results: Results dictionary from evaluate_batch or evaluate_from_dataset
        detailed: If True, print detailed results for each sample
    """
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
        for i, result in enumerate(evaluation_results['detailed_results'][:10]):  # Show first 10
            status = "[PASS]" if result['correct'] else "[FAIL]"
            print(f"{status} Sample {i+1}: Predicted '{result['predicted']}' | "
                  f"Ground Truth '{result['ground_truth']}'")
        if len(evaluation_results['detailed_results']) > 10:
            print(f"... and {len(evaluation_results['detailed_results']) - 10} more samples")
    
    print("=" * 60)

