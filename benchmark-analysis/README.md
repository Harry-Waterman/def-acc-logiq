# Benchmark Analysis

This directory contains comprehensive analysis and visualizations of TinyRod benchmark results.

## Overview

The analysis pipeline processes benchmark JSON files from the `benchmark-results` directory and generates:

- **Comparative visualizations** of model performance
- **Detailed metrics tables** and summaries
- **Repeatability analysis** for consistency evaluation
- **Recommendations** for model selection based on use case

## Files

- `analyze_benchmarks.py` - Main analysis script
- `ANALYSIS_REPORT.md` - Comprehensive text report with insights
- `summary_metrics.csv` - Extracted metrics in CSV format
- `repeatability_metrics.csv` - Repeatability data in CSV format
- `visualizations/` - Directory containing all visualization PNG files

## Usage

### Prerequisites

```bash
pip install pandas matplotlib seaborn numpy
```

### Running the Analysis

```bash
python analyze_benchmarks.py
```

The script will:
1. Load all benchmark JSON files from `../benchmark-results/`
2. Extract metrics and generate visualizations
3. Create a comprehensive analysis report
4. Save all outputs to this directory (visualizations in `visualizations/` subfolder)

## Generated Visualizations

1. **accuracy_comparison.png** - Four-panel comparison showing:
   - Overall accuracy by model
   - Precision vs Recall scatter plot
   - F1 score comparison
   - False Positive Rate vs False Negative Rate

2. **confusion_matrices.png** - Side-by-side confusion matrices for all models

3. **latency_comparison.png** - Inference speed comparison

4. **repeatability_analysis.png** - Consistency metrics including:
   - Score variance by model
   - Reason consistency
   - Distribution histograms

5. **radar_chart.png** - Multi-metric radar chart comparing all models

6. **latency_accuracy_tradeoff.png** - Scatter plot showing the speed/accuracy tradeoff with quadrant labels

7. **quadrant_analysis.png** - Four-quadrant analysis categorizing models by accuracy and latency performance

8. **metrics_heatmap.png** - Comprehensive heatmap showing all metrics (normalized and raw values) across all models

9. **pareto_frontier.png** - Efficiency frontier analysis identifying Pareto-optimal models

## Metrics Explained

- **Accuracy**: Overall percentage of correct predictions
- **Precision**: TP / (TP + FP) - Measures how many flagged emails are actually malicious
- **Recall**: TP / (TP + FN) - Measures how many malicious emails are correctly identified
- **F1 Score**: Harmonic mean of Precision and Recall
- **FPR (False Positive Rate)**: FP / (FP + TN) - Critical for UX (blocking legitimate emails)
- **FNR (False Negative Rate)**: FN / (FN + TP) - Critical for Security (missing threats)
- **Latency**: Average inference time per email

## Model Selection Guidance

The analysis report includes recommendations based on:
- **Production Use**: Best balance of accuracy and low false positive rate
- **High-Security Environments**: Models that minimize false negatives
- **Real-Time Applications**: Fastest inference suitable for browser-based detection

