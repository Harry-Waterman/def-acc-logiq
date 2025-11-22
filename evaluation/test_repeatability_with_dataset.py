"""
Test repeatability benchmark with actual dataset.

This script loads emails from the dataset and tests model repeatability
on real email inputs.
"""

import sys
from pathlib import Path
import pandas as pd
import random

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from evaluation.repeatability import assess_repeatability, benchmark_repeatability, print_repeatability_report
from evaluation.config import DATASET_CONFIG, LABELS, EXAMPLE_REASONS, EVALUATION_CONFIG

# Dataset path from config
DATASET_PATH = parent_dir / DATASET_CONFIG["path"]


def sample_model_with_variation(input_text):
    """
    Sample model function that simulates real model behavior with some variation.
    In production, replace this with your actual SLM model call.
    """
    # Simulate model that is mostly consistent but has occasional variation
    # This could happen due to:
    # - Non-deterministic model behavior
    # - Temperature settings
    # - Model uncertainty
    
    # Simple heuristic: if email contains certain keywords, classify as phishing
    phishing_keywords = ['urgent', 'verify', 'suspended', 'click', 'prize', 'winner', 'free']
    text_lower = input_text.lower()
    
    has_phishing_keywords = any(keyword in text_lower for keyword in phishing_keywords)
    
    # 90% of the time, return consistent result
    # 10% of the time, vary (simulating model uncertainty)
    if random.random() < 0.10:
        # Occasional variation
        if has_phishing_keywords:
            label = LABELS["NOT_MALICIOUS"]  # Wrong classification
            reasons = ["occasional model uncertainty"]
        else:
            label = LABELS["MALICIOUS"]  # False positive
            reasons = ["occasional false positive"]
    else:
        # Consistent classification
        if has_phishing_keywords:
            label = LABELS["MALICIOUS"]
            reasons = EXAMPLE_REASONS[LABELS["MALICIOUS"]][:2]  # Use first 2 from config
        else:
            label = LABELS["NOT_MALICIOUS"]
            reasons = EXAMPLE_REASONS[LABELS["NOT_MALICIOUS"]][:2]  # Use first 2 from config
    
    return {
        "label": label,
        "reasons": reasons
    }


def load_sample_emails(n_samples=5):
    """Load a sample of emails from the dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    # Load balanced sample
    df_full = pd.read_csv(DATASET_PATH)
    
    phishing_samples = df_full[df_full[DATASET_CONFIG['label_column']] == 1][DATASET_CONFIG['text_column']].head(n_samples // 2).tolist()
    legitimate_samples = df_full[df_full[DATASET_CONFIG['label_column']] == 0][DATASET_CONFIG['text_column']].head(n_samples // 2).tolist()
    
    return phishing_samples + legitimate_samples


def main():
    print("=" * 60)
    print("REPEATABILITY BENCHMARK WITH ACTUAL DATASET")
    print("=" * 60)
    
    # Load sample emails
    try:
        print("\nLoading sample emails from dataset...")
        sample_emails = load_sample_emails(n_samples=5)
        print(f"Loaded {len(sample_emails)} sample emails")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Test repeatability for each email
    print("\n" + "=" * 60)
    print("TESTING REPEATABILITY FOR EACH EMAIL")
    print("=" * 60)
    
    for i, email in enumerate(sample_emails):
        print(f"\n{'='*60}")
        print(f"Email {i+1}:")
        print(f"Preview: {email[:100]}...")
        print(f"{'='*60}")
        
        result = assess_repeatability(
            sample_model_with_variation,
            email,
            num_runs=EVALUATION_CONFIG["default_num_runs"]
        )
        
        print_repeatability_report(result, detailed=False)
    
    # Benchmark across all emails
    print("\n" + "=" * 60)
    print("BENCHMARK ACROSS ALL EMAILS")
    print("=" * 60)
    
    benchmark_result = benchmark_repeatability(
        sample_model_with_variation,
        sample_emails,
        num_runs=EVALUATION_CONFIG["default_num_runs"]
    )
    
    print_repeatability_report(benchmark_result, detailed=True)
    
    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    print("""
Consistency Score Interpretation:
- 95-100%: HIGHLY_REPEATABLE - Model is very consistent
- 80-94%:  REPEATABLE - Model is mostly consistent
- 60-79%:  MODERATELY_REPEATABLE - Model has some variation
- <60%:    NOT_REPEATABLE - Model is inconsistent

For production use:
- Aim for >95% consistency score
- Lower entropy indicates more consistent predictions
- If consistency is low, consider:
  * Adjusting model temperature/parameters
  * Using deterministic model settings
  * Post-processing to enforce consistency
    """)


if __name__ == "__main__":
    main()

