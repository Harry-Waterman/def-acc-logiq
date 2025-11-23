# TinyRod Benchmark Analysis Report

## Executive Summary

This report analyzes benchmark results for 7 models tested on phishing email detection.

**Test Configuration:**
- Sample Size: 400 emails (balanced 50/50)
- Temperature: 0.1
- Seed: 42 (for reproducibility)

## Model Performance Rankings

### 🏆 Best Overall Accuracy: qwen_qwen3-8b
- Accuracy: 86.25%
- Precision: 88.77%
- Recall: 83.42%
- F1 Score: 86.01%

### 🎯 Best Precision: qwen_qwen3-4b-2507
- Precision: 90.45% (fewest false positives)

### 🔍 Best Recall: google_gemma-3-4b
- Recall: 85.35% (fewest false negatives)

### ⚖️ Best F1 Score: qwen_qwen3-8b
- F1: 86.01% (balanced precision/recall)

### ⚡ Fastest Inference: qwen_qwen3-4b-2507
- Average Latency: 1.87 seconds

## Detailed Metrics Table

| model | model_full | model_size | accuracy | precision | recall | f1 | fpr | fnr | tp | tn | fp | fn | avg_latency | total | temperature | sample_size | timeout_rate | json_error_rate | error_rate | total_error_rate | success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google_gemma-3-4b | google/gemma-3-4b | 4.0 | 0.7125 | 0.6679841897233202 | 0.8535353535353535 | 0.7494456762749445 | 0.42 | 0.14646464646464646 | 169 | 116 | 84 | 29 | 2.44568052649498 | 400 | 0.1 | 400 | 0.0 | 0.0025 | 0.0025 | 0.005 | 0.995 |
| ibm_granite-4-h-tiny | ibm/granite-4-h-tiny | 4.0 | 0.585 | 0.6098484848484849 | 0.8214285714285714 | 0.7000000000000001 | 0.5852272727272727 | 0.17857142857142858 | 161 | 73 | 103 | 35 | 2.3115645277500154 | 400 | 0.1 | 400 | 0.0 | 0.07 | 0.0 | 0.07 | 0.93 |
| liquid_lfm2-1.2b | liquid/lfm2-1.2b | 1.2 | 0.34 | 0.4095238095238095 | 0.5149700598802395 | 0.4562334217506631 | 0.7126436781609196 | 0.48502994011976047 | 86 | 50 | 124 | 81 | 2.22046317756176 | 400 | 0.1 | 400 | 0.005 | 0.1325 | 0.01 | 0.1475 | 0.8525 |
| phi-4-mini-reasoning | microsoft/phi-4-mini-reasoning | 4.0 | 0.09 | 0.4444444444444444 | 0.26666666666666666 | 0.33333333333333337 | 0.13513513513513514 | 0.7333333333333333 | 4 | 32 | 5 | 11 | 55.48528385519981 | 400 | 0.1 | 400 | 0.7275 | 0.1425 | 0.0 | 0.87 | 0.13 |
| qwen_qwen3-1.7b | qwen/qwen3-1.7b | 1.7 | 0.56 | 0.5869565217391305 | 0.413265306122449 | 0.48502994011976047 | 0.285 | 0.5867346938775511 | 81 | 143 | 57 | 115 | 13.52451581120491 | 400 | 0.1 | 400 | 0.0 | 0.0025 | 0.0075 | 0.01 | 0.99 |
| qwen_qwen3-4b-2507 | qwen/qwen3-4b-2507 | 4.0 | 0.86 | 0.9044943820224719 | 0.8090452261306532 | 0.8541114058355437 | 0.085 | 0.19095477386934673 | 161 | 183 | 17 | 38 | 1.8692622697353363 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-8b | qwen/qwen3-8b | 8.0 | 0.8625 | 0.8877005347593583 | 0.8341708542713567 | 0.8601036269430052 | 0.105 | 0.1658291457286432 | 166 | 179 | 21 | 33 | 2.9322172486782074 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |


## Key Insights

### Accuracy Variation
- Range: 9.00% to 86.25%
- Spread: 77.25%
- Mean: 57.29%

### Error Rate Analysis
- **False Positive Rate (FPR)**: Critical for user experience
  - Lowest: qwen_qwen3-4b-2507 (8.50%)
  - Highest: liquid_lfm2-1.2b (71.26%)
- **False Negative Rate (FNR)**: Critical for security
  - Lowest: google_gemma-3-4b (14.65%)
  - Highest: phi-4-mini-reasoning (73.33%)

### Latency Analysis
- Fastest: qwen_qwen3-4b-2507 (1.87s)
- Slowest: phi-4-mini-reasoning (55.49s)
- Average: 11.54s

## Repeatability Analysis

### Score Consistency
| mean | std |
| --- | --- |
| 0.0 | 0.0 |
| 0.0 | 0.0 |
| 0.0 | nan |
| 0.0 | nan |
| 0.0 | 0.0 |
| 0.0 | 0.0 |
| 18.15 | 25.667976157071674 |


### Reason Consistency
| mean | std |
| --- | --- |
| 1.0 | 0.0 |
| 1.0 | 0.0 |
| 1.0 | nan |
| 1.0 | nan |
| 1.0 | 0.0 |
| 1.0 | 0.0 |
| 1.0 | 0.0 |


### Reason Consistency by Email Type

This analysis shows whether models provide consistent reasoning for benign vs malicious emails:

| Benign | Malicious |
| --- | --- |
| 1.0 | 1.0 |
| 1.0 | 1.0 |
| 1.0 | 0.0 |
| 1.0 | 0.0 |
| 1.0 | 1.0 |
| 1.0 | 1.0 |
| 1.0 | 1.0 |


### Model Size vs Repeatability Correlation

Analysis of whether larger models show different repeatability patterns:

- **Model Size vs Score Variance Correlation**: 0.834

## Recommendations

- **For Production Use**: qwen_qwen3-8b offers the best balance of accuracy with low false positive rate
- **For High-Security Environments**: google_gemma-3-4b minimizes false negatives (missed threats)
- **For Real-Time Applications**: qwen_qwen3-4b-2507 provides the fastest inference suitable for browser-based detection
## Per-Email Statistics Analysis

This section analyzes individual email-level performance patterns:

### WARNING: Timeout/Error Warnings

The following models experienced timeouts or errors during evaluation:

- **google_gemma-3-4b**: 1 JSON parsing errors, 1 other errors out of 400 total (398 successful, 0.5% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **ibm_granite-4-h-tiny**: 28 JSON parsing errors out of 400 total (372 successful, 7.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **liquid_lfm2-1.2b**: 2 timeouts, 53 JSON parsing errors, 4 other errors out of 400 total (341 successful, 14.8% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - Timeouts suggest the model is too slow for production use
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **phi-4-mini-reasoning**: 291 timeouts, 57 JSON parsing errors out of 400 total (52 successful, 87.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - Timeouts suggest the model is too slow for production use
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-1.7b**: 1 JSON parsing errors, 3 other errors out of 400 total (396 successful, 1.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-4b-2507**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-8b**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks


### Key Findings from Per-Email Data:

- **Score Distribution**: Shows model confidence patterns and calibration
  - Well-calibrated models show higher scores for correct predictions
  - Overconfident models show high scores even when incorrect
- **Status Breakdown**: Detailed TP, TN, FP, FN, TIMEOUT, JSON_ERROR, ERROR counts per model
  - Reveals the composition of errors (false positives vs false negatives)
  - TIMEOUT: Model exceeded time limit (typically 60 seconds)
  - JSON_ERROR: Model output was not valid JSON format
  - ERROR: Other types of errors (network, API, etc.)
  - These error cases indicate incomplete evaluations that should be addressed
- **Duration Distribution**: Shows latency variability across emails
  - Identifies if inference time is consistent or has outliers

See `per_email_statistics.png` for detailed visualizations.


## Visualizations

The following visualizations are available:
- `accuracy_comparison.png`: Overall accuracy, precision/recall, F1, and error rates
- `confusion_matrices.png`: Confusion matrices for all models
- `latency_comparison.png`: Inference speed comparison
- `repeatability_analysis.png`: Consistency and repeatability metrics
- `reason_consistency_by_type.png`: Reason consistency broken down by email type (benign vs malicious)
- `model_size_correlation.png`: Correlation between model size and repeatability
- `temperature_impact.png`: Impact of temperature settings on consistency
- `per_email_statistics.png`: Per-email accuracy statistics and distributions
- `radar_chart.png`: Multi-metric radar comparison
- `latency_accuracy_tradeoff.png`: Latency vs accuracy tradeoff scatter plot
- `quadrant_analysis.png`: Four-quadrant analysis categorizing models
- `metrics_heatmap.png`: Comprehensive metrics heatmap
- `pareto_frontier.png`: Pareto efficiency frontier analysis
