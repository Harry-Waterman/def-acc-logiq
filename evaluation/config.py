"""
Configuration constants for evaluation module.

Centralized definitions for labels, reasons, and other constants.
Update these values when integrating with real model outputs.
"""

# Label definitions
LABELS = {
    "MALICIOUS": "malicious",
    "NOT_MALICIOUS": "not_malicious"
}

# Label mapping for dataset normalization
# Maps dataset labels to standard labels
LABEL_MAPPING = {
    # Dataset numeric labels
    0: LABELS["NOT_MALICIOUS"],
    1: LABELS["MALICIOUS"],
    # String variations
    "phishing": LABELS["MALICIOUS"],
    "legitimate": LABELS["NOT_MALICIOUS"],
    "not_malicious": LABELS["NOT_MALICIOUS"],
    "malicious": LABELS["MALICIOUS"]
}

# Example reasons (for testing/demo purposes)
# Replace with actual model output reasons when integrating
EXAMPLE_REASONS = {
    LABELS["MALICIOUS"]: [
        "contains suspicious patterns",
        "potential phishing indicators detected",
        "suspicious link detected",
        "urgent action required without context"
    ],
    LABELS["NOT_MALICIOUS"]: [
        "appears to be legitimate communication",
        "no obvious phishing indicators",
        "normal order update about an existing purchase",
        "no request for money, banking details, or credentials"
    ]
}

# Dataset configuration
DATASET_CONFIG = {
    "path": "dataset/phishing_email.csv",
    "label_column": "label",
    "text_column": "text_combined"
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "default_num_runs": 100,  # For repeatability tests
    "default_sample_size": 20  # For dataset sampling
}

