"""
Repeatability benchmark for SLM model outputs.

This module assesses how consistent/repeatable model classifications are
when given the same input multiple times.
"""

import sys
from pathlib import Path
from typing import Dict, List, Union, Callable
from collections import Counter
import statistics

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from evaluation import evaluate_model_response


def assess_repeatability(
    model_function: Callable,
    test_input: Union[str, Dict],
    num_runs: int = 100,
    label_mapping: Dict[str, str] = None
) -> Dict:
    """
    Assess the repeatability of a model's classification for a given input.
    
    Args:
        model_function: Function that takes input and returns model output dict with 'label' and 'reasons'
        test_input: The input to test (email text or dict)
        num_runs: Number of times to run the model (default: 100)
        label_mapping: Optional mapping to normalize labels
    
    Returns:
        Dictionary with repeatability metrics
    """
    predictions = []
    all_reasons = []
    
    # Run the model multiple times
    for i in range(num_runs):
        try:
            model_output = model_function(test_input)
            
            # Extract label
            if isinstance(model_output, str):
                import json
                model_output = json.loads(model_output)
            
            predicted_label = model_output.get("label", "").lower().strip()
            
            # Apply label mapping if provided
            if label_mapping:
                predicted_label = label_mapping.get(predicted_label, predicted_label)
            
            predictions.append(predicted_label)
            all_reasons.append(model_output.get("reasons", []))
            
        except Exception as e:
            # If model fails, record as error
            predictions.append("ERROR")
            all_reasons.append([f"Model error: {str(e)}"])
    
    # Calculate repeatability metrics
    prediction_counts = Counter(predictions)
    total_predictions = len(predictions)
    
    # Most common prediction
    most_common = prediction_counts.most_common(1)[0] if prediction_counts else (None, 0)
    most_common_label, most_common_count = most_common
    
    # Consistency score (percentage of most common prediction)
    consistency_score = (most_common_count / total_predictions) * 100 if total_predictions > 0 else 0
    
    # Calculate agreement rate (how often the most common prediction appears)
    agreement_rate = consistency_score / 100.0
    
    # Entropy (measure of uncertainty - lower is better for consistency)
    import math
    probabilities = [count / total_predictions for count in prediction_counts.values()]
    entropy = -sum(p * math.log2(p) if p > 0 else 0 for p in probabilities)
    # Normalize entropy (0 = perfectly consistent, 1 = maximum uncertainty)
    # Maximum entropy occurs when all predictions are equally likely
    max_entropy = math.log2(len(prediction_counts)) if len(prediction_counts) > 1 else 0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Variance in predictions (lower is better)
    unique_predictions = len(prediction_counts)
    
    # Distribution of predictions
    prediction_distribution = {
        label: {
            "count": count,
            "percentage": (count / total_predictions) * 100
        }
        for label, count in prediction_counts.items()
    }
    
    # Determine repeatability status
    if consistency_score >= 95:
        status = "HIGHLY_REPEATABLE"
    elif consistency_score >= 80:
        status = "REPEATABLE"
    elif consistency_score >= 60:
        status = "MODERATELY_REPEATABLE"
    else:
        status = "NOT_REPEATABLE"
    
    return {
        "num_runs": num_runs,
        "total_predictions": total_predictions,
        "most_common_prediction": most_common_label,
        "most_common_count": most_common_count,
        "consistency_score": consistency_score,
        "agreement_rate": agreement_rate,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "unique_predictions": unique_predictions,
        "prediction_distribution": prediction_distribution,
        "status": status,
        "all_predictions": predictions,
        "sample_reasons": all_reasons[:5] if all_reasons else []  # First 5 for inspection
    }


def benchmark_repeatability(
    model_function: Callable,
    test_inputs: List[Union[str, Dict]],
    num_runs: int = 100,
    label_mapping: Dict[str, str] = None
) -> Dict:
    """
    Benchmark repeatability across multiple test inputs.
    
    Args:
        model_function: Function that takes input and returns model output
        test_inputs: List of inputs to test
        num_runs: Number of times to run each input (default: 100)
        label_mapping: Optional mapping to normalize labels
    
    Returns:
        Dictionary with aggregate repeatability metrics
    """
    results = []
    
    for i, test_input in enumerate(test_inputs):
        result = assess_repeatability(
            model_function,
            test_input,
            num_runs=num_runs,
            label_mapping=label_mapping
        )
        result["input_index"] = i
        results.append(result)
    
    # Aggregate metrics
    consistency_scores = [r["consistency_score"] for r in results]
    agreement_rates = [r["agreement_rate"] for r in results]
    normalized_entropies = [r["normalized_entropy"] for r in results]
    
    avg_consistency = statistics.mean(consistency_scores) if consistency_scores else 0
    avg_agreement = statistics.mean(agreement_rates) if agreement_rates else 0
    avg_entropy = statistics.mean(normalized_entropies) if normalized_entropies else 0
    
    # Count statuses
    status_counts = Counter([r["status"] for r in results])
    
    return {
        "num_inputs": len(test_inputs),
        "num_runs_per_input": num_runs,
        "total_runs": len(test_inputs) * num_runs,
        "average_consistency_score": avg_consistency,
        "average_agreement_rate": avg_agreement,
        "average_normalized_entropy": avg_entropy,
        "status_distribution": dict(status_counts),
        "individual_results": results
    }


def print_repeatability_report(result: Dict, detailed: bool = False):
    """
    Print a formatted repeatability report.
    
    Args:
        result: Result dictionary from assess_repeatability or benchmark_repeatability
        detailed: If True, print detailed information
    """
    print("=" * 60)
    print("MODEL REPEATABILITY REPORT")
    print("=" * 60)
    
    if "num_inputs" in result:
        # Benchmark report (multiple inputs)
        print(f"\nNumber of Test Inputs: {result['num_inputs']}")
        print(f"Runs per Input: {result['num_runs_per_input']}")
        print(f"Total Runs: {result['total_runs']}")
        print(f"\nAverage Consistency Score: {result['average_consistency_score']:.2f}%")
        print(f"Average Agreement Rate: {result['average_agreement_rate']:.2%}")
        print(f"Average Normalized Entropy: {result['average_normalized_entropy']:.4f}")
        
        print("\nStatus Distribution:")
        for status, count in result['status_distribution'].items():
            print(f"  {status}: {count}")
        
        if detailed and result.get('individual_results'):
            print("\nDetailed Results per Input:")
            print("-" * 60)
            for i, individual in enumerate(result['individual_results'][:10]):  # Show first 10
                print(f"\nInput {i+1}:")
                print(f"  Consistency: {individual['consistency_score']:.2f}%")
                print(f"  Most Common: {individual['most_common_prediction']} ({individual['most_common_count']}/{individual['num_runs']})")
                print(f"  Status: {individual['status']}")
            if len(result['individual_results']) > 10:
                print(f"\n... and {len(result['individual_results']) - 10} more inputs")
    
    else:
        # Single input report
        print(f"\nNumber of Runs: {result['num_runs']}")
        print(f"Most Common Prediction: {result['most_common_prediction']}")
        print(f"  Count: {result['most_common_count']}/{result['total_predictions']}")
        print(f"  Percentage: {result['consistency_score']:.2f}%")
        print(f"\nConsistency Score: {result['consistency_score']:.2f}%")
        print(f"Agreement Rate: {result['agreement_rate']:.2%}")
        print(f"Normalized Entropy: {result['normalized_entropy']:.4f}")
        print(f"Unique Predictions: {result['unique_predictions']}")
        print(f"Status: {result['status']}")
        
        print("\nPrediction Distribution:")
        for label, stats in result['prediction_distribution'].items():
            print(f"  {label}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        if detailed:
            print("\nSample Predictions (first 20):")
            for i, pred in enumerate(result['all_predictions'][:20]):
                print(f"  Run {i+1}: {pred}")
            if len(result['all_predictions']) > 20:
                print(f"  ... and {len(result['all_predictions']) - 20} more")
    
    print("=" * 60)

