# TinyRod Benchmark Analysis Report

## Executive Summary

This report analyzes benchmark results for 5 models tested on phishing email detection.

**Test Configuration:**
- Sample Size: 400 emails (balanced 50/50)
- Temperature: 0.1
- Seed: 42 (for reproducibility)

## Model Performance Rankings

### 🏆 Best Overall Accuracy: qwen_qwen3-4b-2507
- Accuracy: 86.00%
- Precision: 90.45%
- Recall: 80.90%
- F1 Score: 85.41%

### 🎯 Best Precision: qwen_qwen3-4b-2507
- Precision: 90.45% (fewest false positives)

### 🔍 Best Recall: google_gemma-3-4b
- Recall: 85.35% (fewest false negatives)

### ⚖️ Best F1 Score: qwen_qwen3-4b-2507
- F1: 85.41% (balanced precision/recall)

### ⚡ Fastest Inference: qwen_qwen3-4b-2507
- Average Latency: 1.87 seconds

## Detailed Metrics Table

| model | model_full | model_size | accuracy | precision | recall | f1 | fpr | fnr | tp | tn | fp | fn | avg_latency | total | temperature | sample_size |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google_gemma-3-4b | google/gemma-3-4b | 4.0 | 0.7125 | 0.6679841897233202 | 0.8535353535353535 | 0.7494456762749445 | 0.42 | 0.14646464646464646 | 169 | 116 | 84 | 29 | 2.44568052649498 | 400 | 0.1 | 400 |
| ibm_granite-4-h-tiny | ibm/granite-4-h-tiny | 4.0 | 0.585 | 0.6098484848484849 | 0.8214285714285714 | 0.7000000000000001 | 0.5852272727272727 | 0.17857142857142858 | 161 | 73 | 103 | 35 | 2.3115645277500154 | 400 | 0.1 | 400 |
| liquid_lfm2-1.2b | liquid/lfm2-1.2b | 1.2 | 0.34 | 0.4095238095238095 | 0.5149700598802395 | 0.4562334217506631 | 0.7126436781609196 | 0.48502994011976047 | 86 | 50 | 124 | 81 | 2.22046317756176 | 400 | 0.1 | 400 |
| qwen_qwen3-1.7b | qwen/qwen3-1.7b | 1.7 | 0.56 | 0.5869565217391305 | 0.413265306122449 | 0.48502994011976047 | 0.285 | 0.5867346938775511 | 81 | 143 | 57 | 115 | 13.52451581120491 | 400 | 0.1 | 400 |
| qwen_qwen3-4b-2507 | qwen/qwen3-4b-2507 | 4.0 | 0.86 | 0.9044943820224719 | 0.8090452261306532 | 0.8541114058355437 | 0.085 | 0.19095477386934673 | 161 | 183 | 17 | 38 | 1.8692622697353363 | 400 | 0.1 | 400 |


## Key Insights

### Accuracy Variation
- Range: 34.00% to 86.00%
- Spread: 52.00%
- Mean: 61.15%

### Error Rate Analysis
- **False Positive Rate (FPR)**: Critical for user experience
  - Lowest: qwen_qwen3-4b-2507 (8.50%)
  - Highest: liquid_lfm2-1.2b (71.26%)
- **False Negative Rate (FNR)**: Critical for security
  - Lowest: google_gemma-3-4b (14.65%)
  - Highest: qwen_qwen3-1.7b (58.67%)

### Latency Analysis
- Fastest: qwen_qwen3-4b-2507 (1.87s)
- Slowest: qwen_qwen3-1.7b (13.52s)
- Average: 4.47s

## Repeatability Analysis

### Score Consistency
| mean | std |
| --- | --- |
| 0.0 | 0.0 |
| 0.0 | 0.0 |
| 0.0 | nan |
| 0.0 | 0.0 |
| 0.0 | 0.0 |


### Reason Consistency
| mean | std |
| --- | --- |
| 1.0 | 0.0 |
| 1.0 | 0.0 |
| 1.0 | nan |
| 1.0 | 0.0 |
| 1.0 | 0.0 |


### Reason Consistency by Email Type

This analysis shows whether models provide consistent reasoning for benign vs malicious emails:

| Benign | Malicious |
| --- | --- |
| 1.0 | 1.0 |
| 1.0 | 1.0 |
| 1.0 | 0.0 |
| 1.0 | 1.0 |
| 1.0 | 1.0 |


### Model Size vs Repeatability Correlation

Analysis of whether larger models show different repeatability patterns:


## Recommendations

- **For Production Use**: qwen_qwen3-4b-2507 offers the best balance of accuracy with low false positive rate
- **For High-Security Environments**: google_gemma-3-4b minimizes false negatives (missed threats)
- **For Real-Time Applications**: qwen_qwen3-4b-2507 provides the fastest inference suitable for browser-based detection
## Per-Email Statistics Analysis

This section analyzes individual email-level performance patterns:

### Key Findings from Per-Email Data:

- **Score Distribution**: Shows model confidence patterns and calibration
  - Well-calibrated models show higher scores for correct predictions
  - Overconfident models show high scores even when incorrect
- **Status Breakdown**: Detailed TP, TN, FP, FN counts per model
  - Reveals the composition of errors (false positives vs false negatives)
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
