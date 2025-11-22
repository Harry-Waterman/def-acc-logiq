"""
Test repeatability benchmark with sample model outputs.

This script demonstrates how to use the repeatability benchmark
to assess model consistency.
"""

import sys
from pathlib import Path
import random

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from evaluation.repeatability import assess_repeatability, benchmark_repeatability, print_repeatability_report


def sample_model_function_consistent(input_data):
    """
    Sample model function that is highly consistent (always returns same result).
    This simulates a well-behaved model.
    """
    # Always return the same prediction for the same input
    # In reality, this would call your actual SLM model
    return {
        "label": "not_malicious",
        "reasons": [
            "consistent classification",
            "no variation in output"
        ]
    }


def sample_model_function_inconsistent(input_data):
    """
    Sample model function that is inconsistent (randomly varies).
    This simulates a model with poor repeatability.
    """
    # Randomly vary predictions (bad model behavior)
    if random.random() < 0.5:
        label = "malicious"
        reasons = ["suspicious patterns detected"]
    else:
        label = "not_malicious"
        reasons = ["appears legitimate"]
    
    return {
        "label": label,
        "reasons": reasons
    }


def sample_model_function_mostly_consistent(input_data):
    """
    Sample model function that is mostly consistent (95% of the time).
    This simulates a good but not perfect model.
    """
    # 95% of the time return same, 5% vary
    if random.random() < 0.95:
        label = "not_malicious"
        reasons = ["legitimate communication"]
    else:
        label = "malicious"
        reasons = ["occasional false positive"]
    
    return {
        "label": label,
        "reasons": reasons
    }


def main():
    print("=" * 60)
    print("REPEATABILITY BENCHMARK TEST")
    print("=" * 60)
    
    # Test input (sample email)
    test_email = "Your account has been suspended. Click here to verify: http://suspicious-link.com"
    
    # Test 1: Highly consistent model
    print("\n" + "=" * 60)
    print("TEST 1: Highly Consistent Model")
    print("=" * 60)
    result1 = assess_repeatability(
        sample_model_function_consistent,
        test_email,
        num_runs=100
    )
    print_repeatability_report(result1, detailed=True)
    
    # Test 2: Inconsistent model
    print("\n" + "=" * 60)
    print("TEST 2: Inconsistent Model (Random)")
    print("=" * 60)
    result2 = assess_repeatability(
        sample_model_function_inconsistent,
        test_email,
        num_runs=100
    )
    print_repeatability_report(result2, detailed=True)
    
    # Test 3: Mostly consistent model
    print("\n" + "=" * 60)
    print("TEST 3: Mostly Consistent Model (95%)")
    print("=" * 60)
    result3 = assess_repeatability(
        sample_model_function_mostly_consistent,
        test_email,
        num_runs=100
    )
    print_repeatability_report(result3, detailed=True)
    
    # Test 4: Benchmark across multiple inputs
    print("\n" + "=" * 60)
    print("TEST 4: Benchmark Across Multiple Inputs")
    print("=" * 60)
    
    test_inputs = [
        "Your account has been suspended. Click here to verify.",
        "Thank you for your order. Your package will arrive soon.",
        "URGENT: Verify your account now or it will be deleted!",
        "Meeting scheduled for tomorrow at 2 PM.",
        "You have won $1,000,000! Claim your prize now!"
    ]
    
    benchmark_result = benchmark_repeatability(
        sample_model_function_mostly_consistent,
        test_inputs,
        num_runs=50  # Fewer runs per input for faster testing
    )
    print_repeatability_report(benchmark_result, detailed=True)
    
    print("\n" + "=" * 60)
    print("REPEATABILITY TESTING COMPLETE")
    print("=" * 60)
    print("\nTo use with your actual SLM model:")
    print("1. Replace sample_model_function with your actual model function")
    print("2. The function should take input and return dict with 'label' and 'reasons'")
    print("3. Adjust num_runs parameter as needed (default: 100)")
    print("4. Review consistency_score - higher is better (aim for >95%)")


if __name__ == "__main__":
    main()
