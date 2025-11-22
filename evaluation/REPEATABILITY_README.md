# Repeatability Benchmark

This module assesses the repeatability/consistency of SLM model classifications. It tests whether the model gives consistent outputs when given the same input multiple times.

## Overview

A repeatable model should classify the same email the same way every time (or nearly every time). This benchmark helps identify:
- Models that are inconsistent (e.g., classify as phishing 50% of the time, not phishing 50% of the time)
- Models that are highly consistent (e.g., classify as phishing 99/100 times for the same input)
- Variation patterns in model outputs

## Usage

### Basic Example

```python
from evaluation.repeatability import assess_repeatability, print_repeatability_report

def your_model_function(input_text):
    """Your actual SLM model call"""
    # Replace with your actual model
    return {
        "label": "malicious",  # or "not_malicious"
        "reasons": ["reason 1", "reason 2"]
    }

# Test repeatability for a single input
result = assess_repeatability(
    your_model_function,
    "Your account has been suspended. Click here to verify.",
    num_runs=100  # Run 100 times (configurable)
)

print_repeatability_report(result, detailed=True)
```

### Benchmark Multiple Inputs

```python
from evaluation.repeatability import benchmark_repeatability

test_emails = [
    "Your account has been suspended.",
    "Thank you for your order.",
    "URGENT: Verify your account now!"
]

result = benchmark_repeatability(
    your_model_function,
    test_emails,
    num_runs=100
)

print_repeatability_report(result, detailed=True)
```

## Metrics

### Consistency Score
- **Range**: 0-100%
- **Meaning**: Percentage of times the most common prediction appears
- **Good**: >95% (highly repeatable)
- **Bad**: <60% (not repeatable)

### Agreement Rate
- **Range**: 0.0-1.0
- **Meaning**: Same as consistency score, normalized to 0-1
- **Interpretation**: Higher is better

### Normalized Entropy
- **Range**: 0.0-1.0
- **Meaning**: Measure of uncertainty in predictions
- **Good**: <0.1 (low uncertainty, high consistency)
- **Bad**: >0.5 (high uncertainty, low consistency)

### Status Categories
- **HIGHLY_REPEATABLE**: 95-100% consistency
- **REPEATABLE**: 80-94% consistency
- **MODERATELY_REPEATABLE**: 60-79% consistency
- **NOT_REPEATABLE**: <60% consistency

## Configuration

### Number of Runs
The `num_runs` parameter controls how many times to test each input:
- **Default**: 100
- **More runs**: More accurate but slower
- **Fewer runs**: Faster but less reliable

```python
# Test with 200 runs for more accuracy
result = assess_repeatability(
    model_function,
    input_text,
    num_runs=200
)
```

### Label Mapping
If your model uses different label names, use label mapping:

```python
label_mapping = {
    "phishing": "malicious",
    "legitimate": "not_malicious"
}

result = assess_repeatability(
    model_function,
    input_text,
    num_runs=100,
    label_mapping=label_mapping
)
```

## Example Output

```
============================================================
MODEL REPEATABILITY REPORT
============================================================

Number of Runs: 100
Most Common Prediction: not_malicious
  Count: 98/100
  Percentage: 98.00%

Consistency Score: 98.00%
Agreement Rate: 98.00%
Normalized Entropy: 0.1414
Unique Predictions: 2
Status: HIGHLY_REPEATABLE

Prediction Distribution:
  not_malicious: 98 (98.00%)
  malicious: 2 (2.00%)
============================================================
```

## Test Scripts

### Basic Test
```bash
python evaluation/test_repeatability.py
```
Tests with sample model functions (consistent, inconsistent, mostly consistent).

### Test with Dataset
```bash
python evaluation/test_repeatability_with_dataset.py
```
Tests with actual emails from the phishing email dataset.

## Integration with Your Model

To use with your actual SLM model:

1. **Create a wrapper function**:
```python
def my_slm_model(input_text):
    # Call your actual SLM model
    response = your_slm_api_call(input_text)
    
    # Parse response to match expected format
    return {
        "label": response["label"],  # "malicious" or "not_malicious"
        "reasons": response.get("reasons", [])
    }
```

2. **Run repeatability test**:
```python
from evaluation.repeatability import assess_repeatability, print_repeatability_report

result = assess_repeatability(
    my_slm_model,
    "test email text",
    num_runs=100
)

print_repeatability_report(result)
```

3. **Interpret results**:
   - If consistency < 95%, consider:
     - Adjusting model temperature/parameters
     - Using deterministic settings
     - Post-processing to enforce consistency

## Best Practices

1. **Test with representative inputs**: Use real emails from your dataset
2. **Run sufficient tests**: Use at least 100 runs per input
3. **Monitor consistency**: Track consistency scores over time
4. **Set thresholds**: Define minimum acceptable consistency (e.g., >95%)
5. **Investigate variations**: If consistency is low, investigate why the model varies

## Troubleshooting

**Low consistency scores:**
- Check if model has non-deterministic settings (temperature, sampling)
- Verify model is not using random seed variations
- Consider using deterministic model parameters

**Model errors during testing:**
- Errors are recorded as "ERROR" predictions
- Check model API stability and error handling
- Ensure model can handle the same input multiple times

