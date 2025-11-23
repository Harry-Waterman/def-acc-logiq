# TinyRod Benchmark Analysis Report

## Executive Summary

This report analyzes benchmark results for 16 models tested on phishing email detection.

**Test Configuration:**
- Sample Size: 400 emails (balanced 50/50)
- Temperature: 0.1
- Seed: 42 (for reproducibility)

## Model Performance Rankings

### 🏆 Best Overall Accuracy: [CLOUD]-gpt-5.1-0.1-temp
- Accuracy: 91.75%
- Precision: 99.41%
- Sensitivity (Recall): 84.00%
- F1 Score: 91.06%

### 🎯 Best Precision: [CLOUD]-gpt-5.1-0.1-temp
- Precision: 99.41% (fewest false positives)

### 🔍 Best Sensitivity (Recall): google_gemma-3-4b-0.7-temp
- Sensitivity (Recall): 86.29% (fewest false negatives)

### ⚖️ Best F1 Score: [CLOUD]-gpt-5.1-0.1-temp
- F1: 91.06% (balanced precision/recall)

### ⚡ Fastest Inference: qwen_qwen3-1.7b-0.1-temp
- Average Latency: 0.79 seconds

## Detailed Metrics Table

| model | model_full | model_size | accuracy | precision | recall | f1 | fpr | fnr | tp | tn | fp | fn | avg_latency | total | temperature | sample_size | timeout_rate | json_error_rate | error_rate | total_error_rate | success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google_gemma-3-4b-0.1-temp | google/gemma-3-4b | 4.0 | 0.7125 | 0.6679841897233202 | 0.8535353535353535 | 0.7494456762749445 | 0.42 | 0.14646464646464646 | 169 | 116 | 84 | 29 | 2.44568052649498 | 400 | 0.1 | 400 | 0.0 | 0.0025 | 0.0025 | 0.005 | 0.995 |
| google_gemma-3-4b-0.7-temp | google/gemma-3-4b | 4.0 | 0.715 | 0.6746031746031746 | 0.8629441624365483 | 0.757238307349666 | 0.41414141414141414 | 0.13705583756345177 | 170 | 116 | 82 | 27 | 3.687254260182381 | 400 | 0.7 | 400 | 0.0 | 0.01 | 0.0025 | 0.0125 | 0.9875 |
| ibm_granite-4-h-tiny-0.1-temp | ibm/granite-4-h-tiny | 4.0 | 0.585 | 0.6098484848484849 | 0.8214285714285714 | 0.7000000000000001 | 0.5852272727272727 | 0.17857142857142858 | 161 | 73 | 103 | 35 | 2.3115645277500154 | 400 | 0.1 | 400 | 0.0 | 0.07 | 0.0 | 0.07 | 0.93 |
| ibm_granite-4-h-tiny-0.7-temp | ibm/granite-4-h-tiny | 4.0 | 0.535 | 0.5875912408759124 | 0.8429319371727748 | 0.6924731182795698 | 0.6807228915662651 | 0.15706806282722513 | 161 | 53 | 113 | 30 | 5.454019554257393 | 400 | 0.7 | 400 | 0.0 | 0.1075 | 0.0 | 0.1075 | 0.8925 |
| liquid_lfm2-1.2b-0.1-temp | liquid/lfm2-1.2b | 1.2 | 0.34 | 0.4095238095238095 | 0.5149700598802395 | 0.4562334217506631 | 0.7126436781609196 | 0.48502994011976047 | 86 | 50 | 124 | 81 | 2.22046317756176 | 400 | 0.1 | 400 | 0.005 | 0.1325 | 0.01 | 0.1475 | 0.8525 |
| liquid_lfm2-1.2b-0.7-temp | liquid/lfm2-1.2b | 1.2 | 0.3125 | 0.4666666666666667 | 0.55 | 0.5049180327868854 | 0.6470588235294118 | 0.45 | 77 | 48 | 88 | 63 | 3.646066211462021 | 400 | 0.7 | 400 | 0.005 | 0.3 | 0.005 | 0.31 | 0.69 |
| phi-4-mini-reasoning-0.1-temp | microsoft/phi-4-mini-reasoning | 4.0 | 0.09 | 0.4444444444444444 | 0.26666666666666666 | 0.33333333333333337 | 0.13513513513513514 | 0.7333333333333333 | 4 | 32 | 5 | 11 | 55.48528385519981 | 400 | 0.1 | 400 | 0.7275 | 0.1425 | 0.0 | 0.87 | 0.13 |
| qwen_qwen3-1.7b-0.1-temp-reasoning | qwen/qwen3-1.7b | 1.7 | 0.56 | 0.5869565217391305 | 0.413265306122449 | 0.48502994011976047 | 0.285 | 0.5867346938775511 | 81 | 143 | 57 | 115 | 13.52451581120491 | 400 | 0.1 | 400 | 0.0 | 0.0025 | 0.0075 | 0.01 | 0.99 |
| qwen_qwen3-1.7b-0.1-temp | qwen/qwen3-1.7b | 1.7 | 0.49 | 0.0 | 0.0 | 0.0 | 0.02 | 1.0 | 0 | 196 | 4 | 199 | 0.7858496206998825 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-1.7b-0.7-temp | qwen/qwen3-1.7b | 1.7 | 0.4975 | 0.4 | 0.010050251256281407 | 0.019607843137254905 | 0.015 | 0.9899497487437185 | 2 | 197 | 3 | 197 | 1.3374188286066055 | 400 | 0.7 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-4b-2507-0.1-temp | qwen/qwen3-4b-2507 | 4.0 | 0.86 | 0.9044943820224719 | 0.8090452261306532 | 0.8541114058355437 | 0.085 | 0.19095477386934673 | 161 | 183 | 17 | 38 | 1.8692622697353363 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-4b-2507-0.7-temp | qwen/qwen3-4b-2507 | 4.0 | 0.8525 | 0.8983050847457628 | 0.7989949748743719 | 0.8457446808510639 | 0.09 | 0.20100502512562815 | 159 | 182 | 18 | 40 | 2.722102379202843 | 400 | 0.7 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-8b-0.1-temp | qwen/qwen3-8b | 8.0 | 0.8625 | 0.8877005347593583 | 0.8341708542713567 | 0.8601036269430052 | 0.105 | 0.1658291457286432 | 166 | 179 | 21 | 33 | 2.9322172486782074 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0025 | 0.0025 | 0.9975 |
| qwen_qwen3-8b-0.7-temp | qwen/qwen3-8b | 8.0 | 0.8425 | 0.8624338624338624 | 0.8316326530612245 | 0.8467532467532467 | 0.13 | 0.1683673469387755 | 163 | 174 | 26 | 33 | 3.7322749555110932 | 400 | 0.7 | 400 | 0.005 | 0.0 | 0.005 | 0.01 | 0.99 |
| [CLOUD]-gpt-5.1-0.1-temp | gpt-5.1-2025-11-13 | 5.1 | 0.9175 | 0.9940828402366864 | 0.84 | 0.9105691056910568 | 0.005 | 0.16 | 168 | 199 | 1 | 32 | 1.4917605406045913 | 400 | 0.1 | 400 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |
| [CLOUD]-gpt-5.1-0.7-temp | gpt-5.1-2025-11-13 | 5.1 | 0.9075 | 0.9822485207100592 | 0.83 | 0.8997289972899729 | 0.015 | 0.17 | 166 | 197 | 3 | 34 | 1.4758550786972047 | 400 | 0.7 | 400 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |


## Key Insights

### Accuracy Variation
- Range: 9.00% to 91.75%
- Spread: 82.75%
- Mean: 63.00%

### Error Rate Analysis
- **False Positive Rate (FPR)**: Critical for user experience
  - Lowest: [CLOUD]-gpt-5.1-0.1-temp (0.50%)
  - Highest: liquid_lfm2-1.2b-0.1-temp (71.26%)
- **False Negative Rate (FNR)**: Critical for security
  - Lowest: google_gemma-3-4b-0.7-temp (13.71%)
  - Highest: qwen_qwen3-1.7b-0.1-temp (100.00%)

### Latency Analysis
- Fastest: qwen_qwen3-1.7b-0.1-temp (0.79s)
- Slowest: phi-4-mini-reasoning-0.1-temp (55.49s)
- Average: 6.57s

## Repeatability Analysis

### Score Consistency
| mean | std |
| --- | --- |
| 0.33454545454545453 | 0.4731187190484536 |
| 24.988434343434342 | 35.33898275095447 |
| 76.19542928467413 | 52.41781791410515 |
| 1094.9761904761904 | 516.6930265384558 |
| 40.5 | 57.27564927611035 |
| 0.7121212121212122 | 1.007091476235386 |
| 18.97050505050505 | 16.31459621530368 |


### Reason Consistency
| mean | std |
| --- | --- |
| 1.0 | 0.0 |
| 0.765 | 0.3323401871576773 |
| 0.04997176736307171 | 0.05230434123965992 |
| 0.6190476190476191 | 0.06734350297014734 |
| 1.0 | 0.0 |
| 1.0 | 0.0 |
| 1.0 | 0.0 |


### Reason Consistency by Email Type

This analysis shows whether models provide consistent reasoning for benign vs malicious emails:

| Benign | Malicious |
| --- | --- |
| 1.0 | 1.0 |
| 1.0 | 0.53 |
| 0.012987012987012988 | 0.08695652173913043 |
| 0.5714285714285714 | 0.6666666666666666 |
| 1.0 | 1.0 |
| 1.0 | 1.0 |
| 1.0 | 1.0 |


### Model Size vs Repeatability Correlation

Analysis of whether larger models show different repeatability patterns:

- **Model Size vs Score Variance Correlation**: -0.080
- **Model Size vs Reason Consistency Correlation**: 0.735

## Recommendations

- **For Production Use**: [CLOUD]-gpt-5.1-0.1-temp offers the best balance of accuracy with low false positive rate
- **For High-Security Environments**: google_gemma-3-4b-0.7-temp minimizes false negatives (missed threats)
- **For Real-Time Applications**: qwen_qwen3-1.7b-0.1-temp provides the fastest inference suitable for browser-based detection
## Per-Email Statistics Analysis

This section analyzes individual email-level performance patterns:

### WARNING: Timeout/Error Warnings

The following models experienced timeouts or errors during evaluation:

- **google_gemma-3-4b-0.1-temp**: 1 JSON parsing errors, 1 other errors out of 400 total (398 successful, 0.5% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **google_gemma-3-4b-0.7-temp**: 4 JSON parsing errors, 1 other errors out of 400 total (395 successful, 1.2% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **ibm_granite-4-h-tiny-0.1-temp**: 28 JSON parsing errors out of 400 total (372 successful, 7.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **ibm_granite-4-h-tiny-0.7-temp**: 43 JSON parsing errors out of 400 total (357 successful, 10.8% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **liquid_lfm2-1.2b-0.1-temp**: 2 timeouts, 53 JSON parsing errors, 4 other errors out of 400 total (341 successful, 14.8% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - Timeouts suggest the model is too slow for production use
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **liquid_lfm2-1.2b-0.7-temp**: 2 timeouts, 120 JSON parsing errors, 2 other errors out of 400 total (276 successful, 31.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - Timeouts suggest the model is too slow for production use
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **phi-4-mini-reasoning-0.1-temp**: 291 timeouts, 57 JSON parsing errors out of 400 total (52 successful, 87.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - Timeouts suggest the model is too slow for production use
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-1.7b-0.1-temp-reasoning**: 1 JSON parsing errors, 3 other errors out of 400 total (396 successful, 1.0% failed)
  - JSON parsing errors indicate the model output was not valid JSON format
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-1.7b-0.1-temp**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-1.7b-0.7-temp**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-4b-2507-0.1-temp**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-4b-2507-0.7-temp**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-8b-0.1-temp**: 1 other errors out of 400 total (399 successful, 0.2% failed)
  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading
  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks

- **qwen_qwen3-8b-0.7-temp**: 2 timeouts, 2 other errors out of 400 total (396 successful, 1.0% failed)
  - Timeouts suggest the model is too slow for production use
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

See `visualizations/per_email/` for detailed visualizations.


## Visualizations

Visualizations are organized into folders by category:

### Accuracy Metrics (`visualizations/accuracy/`)
- `accuracy_bar.png`: Overall accuracy comparison
- `precision_vs_recall.png`: Precision vs Sensitivity (Recall) scatter plot
- `f1_score_bar.png`: F1 score comparison
- `fpr_vs_fnr.png`: False Positive Rate vs False Negative Rate

### Repeatability Analysis (`visualizations/repeatability/`)
- `score_variance_bar.png`: Score variance by model (lower is better)
- `reason_consistency_bar.png`: Reason consistency by model (higher is better)
- `score_mean_by_type.png`: Score mean by model and email type
- `variance_vs_consistency.png`: Consistency trade-off scatter plot
- `consistency_by_type.png`: Reason consistency by email type
- `variance_by_type.png`: Score variance by email type

### Per-Email Statistics (`visualizations/per_email/`)
- `score_dist_correct_incorrect.png`: Score distribution for correct vs incorrect predictions
- `score_dist_by_model.png`: Score distribution by model
- `status_breakdown.png`: Prediction status breakdown (TP, TN, FP, FN, errors)
- `duration_distribution.png`: Inference duration distribution by model

### Temperature Impact (`visualizations/temperature/`)
- `temperature_vs_accuracy.png`: Temperature impact on accuracy
- `temperature_vs_precision.png`: Temperature impact on precision
- `temperature_vs_recall.png`: Temperature impact on sensitivity (recall)
- `temperature_vs_f1.png`: Temperature impact on F1 score
- `temperature_vs_fpr.png`: Temperature impact on false positive rate
- `temperature_vs_fnr.png`: Temperature impact on false negative rate

### Other Visualizations (`visualizations/`)
- `confusion_matrices.png`: Confusion matrices for all models
- `latency_comparison.png`: Inference speed comparison
- `radar_chart.png`: Multi-metric radar comparison
- `latency_accuracy_tradeoff.png`: Latency vs accuracy tradeoff scatter plot
- `quadrant_analysis.png`: Four-quadrant analysis categorizing models
- `metrics_heatmap.png`: Comprehensive metrics heatmap
- `pareto_frontier.png`: Pareto efficiency frontier analysis
