"""
Test the repeatability benchmark with the Nigerian Fraud email dataset.

Loads sender/receiver/date/subject/body/urls/label columns and assesses
how consistently the sample model classifies the same email input.
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
EMAIL_FIELDS = DATASET_CONFIG.get("email_fields", [])
TEXT_FIELDS = DATASET_CONFIG.get("text_fields") or [DATASET_CONFIG["text_column"]]


def build_email_text(row):
    """Combine relevant text fields into a single string for model input."""
    parts = []
    for field in TEXT_FIELDS:
        if field in row and pd.notna(row[field]):
            value = str(row[field]).strip()
            if value:
                parts.append(value)
    if not parts and DATASET_CONFIG["text_column"] in row and pd.notna(row[DATASET_CONFIG["text_column"]]):
        parts.append(str(row[DATASET_CONFIG["text_column"]]).strip())
    return " ".join(parts)


def print_email_metadata(record):
    """Pretty-print configured email fields for repeatability samples."""
    for field in EMAIL_FIELDS:
        value = record.get(field, "")
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        if field == "body" and len(value_str) > 120:
            value_str = f"{value_str[:120]}..."
        print(f"  {field.title()}: {value_str}")


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
    """Load a sample of emails (with metadata) from the dataset."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    # Load balanced sample
    df_full = pd.read_csv(DATASET_PATH)
    label_col = DATASET_CONFIG['label_column']
    half = max(1, n_samples // 2)
    phishing_rows = df_full[df_full[label_col] == 1].head(half)
    legitimate_rows = df_full[df_full[label_col] == 0].head(half)
    
    if len(phishing_rows) == 0 or len(legitimate_rows) == 0:
        print("Warning: Could not find balanced classes; sampling consecutive rows instead.")
        samples = df_full.head(n_samples).reset_index(drop=True)
    else:
        samples = pd.concat([phishing_rows, legitimate_rows]).sample(frac=1).reset_index(drop=True)
    
    records = []
    for _, row in samples.iterrows():
        record = {field: row.get(field, "") for field in EMAIL_FIELDS if field in row}
        record[label_col] = row[label_col]
        record["text"] = build_email_text(row)
        records.append(record)
    
    return records


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
        print_email_metadata(email)
        print(f"Preview: {email['text'][:100]}...")
        print(f"{'='*60}")
        
        result = assess_repeatability(
            sample_model_with_variation,
            email["text"],
            num_runs=EVALUATION_CONFIG["default_num_runs"]
        )
        
        print_repeatability_report(result, detailed=False)
    
    # Benchmark across all emails
    print("\n" + "=" * 60)
    print("BENCHMARK ACROSS ALL EMAILS")
    print("=" * 60)
    
    benchmark_result = benchmark_repeatability(
        sample_model_with_variation,
        [email["text"] for email in sample_emails],
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

