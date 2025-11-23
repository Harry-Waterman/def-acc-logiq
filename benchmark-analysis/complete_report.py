#!/usr/bin/env python3
"""Complete the analysis report with all sections."""
import json
import os
import glob
from pathlib import Path
import pandas as pd

def load_benchmark_results(results_dir: str = "../benchmark-results"):
    """Load all benchmark JSON files."""
    results = []
    json_files = glob.glob(os.path.join(results_dir, "benchmark_results_*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                filename = os.path.basename(file_path)
                model_name = filename.replace("benchmark_results_", "").replace("_seed42.json", "")
                data['model_name'] = model_name
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def extract_summary_metrics(results):
    """Extract summary metrics."""
    metrics = []
    for result in results:
        config = result.get('config', {})
        accuracy = result.get('accuracy_results', {})
        metrics.append({
            'model': result.get('model_name', 'unknown'),
            'model_full': config.get('model', 'unknown'),
            'accuracy': accuracy.get('accuracy', 0),
            'precision': accuracy.get('precision', 0),
            'recall': accuracy.get('recall', 0),
            'f1': accuracy.get('f1', 0),
            'fpr': accuracy.get('fpr', 0),
            'fnr': accuracy.get('fnr', 0),
            'tp': accuracy.get('tp', 0),
            'tn': accuracy.get('tn', 0),
            'fp': accuracy.get('fp', 0),
            'fn': accuracy.get('fn', 0),
            'avg_latency': accuracy.get('avg_latency', 0),
            'total': accuracy.get('total', 0),
            'temperature': config.get('temperature', 0),
            'sample_size': config.get('sample_size', 0),
        })
    return pd.DataFrame(metrics)

def extract_repeatability_metrics(results):
    """Extract repeatability metrics."""
    repeatability_data = []
    for result in results:
        model_name = result.get('model_name', 'unknown')
        repeatability = result.get('repeatability_results', [])
        for item in repeatability:
            repeatability_data.append({
                'model': model_name,
                'ground_truth': item.get('ground_truth', 0),
                'score_mean': item.get('score_mean', 0),
                'score_variance': item.get('score_variance', 0),
                'reason_consistency': item.get('reason_consistency', 0),
            })
    return pd.DataFrame(repeatability_data)

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table."""
    if df.empty:
        return ""
    headers = list(df.columns)
    markdown = "| " + " | ".join(str(h) for h in headers) + " |\n"
    markdown += "| " + " | ".join("---" for _ in headers) + " |\n"
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(row[col]) for col in headers) + " |\n"
    return markdown

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark-results"
    output_dir = script_dir
    
    print("Loading benchmark results...")
    results = load_benchmark_results(str(results_dir))
    
    if not results:
        print("No benchmark results found!")
        exit(1)
    
    df = extract_summary_metrics(results)
    repeat_df = extract_repeatability_metrics(results)
    
    # Generate complete report
    report_path = os.path.join(output_dir, 'ANALYSIS_REPORT.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# TinyRod Benchmark Analysis Report\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"This report analyzes benchmark results for {len(df)} models tested on phishing email detection.\n\n")
        f.write(f"**Test Configuration:**\n")
        f.write(f"- Sample Size: {df['sample_size'].iloc[0]} emails (balanced 50/50)\n")
        f.write(f"- Temperature: {df['temperature'].iloc[0]}\n")
        f.write(f"- Seed: 42 (for reproducibility)\n\n")
        
        f.write("## Model Performance Rankings\n\n")
        
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        f.write(f"### 🏆 Best Overall Accuracy: {best_accuracy['model']}\n")
        f.write(f"- Accuracy: {best_accuracy['accuracy']:.2%}\n")
        f.write(f"- Precision: {best_accuracy['precision']:.2%}\n")
        f.write(f"- Recall: {best_accuracy['recall']:.2%}\n")
        f.write(f"- F1 Score: {best_accuracy['f1']:.2%}\n\n")
        
        best_precision = df.loc[df['precision'].idxmax()]
        f.write(f"### 🎯 Best Precision: {best_precision['model']}\n")
        f.write(f"- Precision: {best_precision['precision']:.2%} (fewest false positives)\n\n")
        
        best_recall = df.loc[df['recall'].idxmax()]
        f.write(f"### 🔍 Best Recall: {best_recall['model']}\n")
        f.write(f"- Recall: {best_recall['recall']:.2%} (fewest false negatives)\n\n")
        
        best_f1 = df.loc[df['f1'].idxmax()]
        f.write(f"### ⚖️ Best F1 Score: {best_f1['model']}\n")
        f.write(f"- F1: {best_f1['f1']:.2%} (balanced precision/recall)\n\n")
        
        fastest = df.loc[df['avg_latency'].idxmin()]
        f.write(f"### ⚡ Fastest Inference: {fastest['model']}\n")
        f.write(f"- Average Latency: {fastest['avg_latency']:.2f} seconds\n\n")
        
        f.write("## Detailed Metrics Table\n\n")
        f.write(dataframe_to_markdown(df))
        f.write("\n\n")
        
        f.write("## Key Insights\n\n")
        
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        f.write(f"### Accuracy Variation\n")
        f.write(f"- Range: {df['accuracy'].min():.2%} to {df['accuracy'].max():.2%}\n")
        f.write(f"- Spread: {acc_range:.2%}\n")
        f.write(f"- Mean: {df['accuracy'].mean():.2%}\n\n")
        
        f.write(f"### Error Rate Analysis\n")
        f.write(f"- **False Positive Rate (FPR)**: Critical for user experience\n")
        f.write(f"  - Lowest: {df.loc[df['fpr'].idxmin(), 'model']} ({df['fpr'].min():.2%})\n")
        f.write(f"  - Highest: {df.loc[df['fpr'].idxmax(), 'model']} ({df['fpr'].max():.2%})\n")
        f.write(f"- **False Negative Rate (FNR)**: Critical for security\n")
        f.write(f"  - Lowest: {df.loc[df['fnr'].idxmin(), 'model']} ({df['fnr'].min():.2%})\n")
        f.write(f"  - Highest: {df.loc[df['fnr'].idxmax(), 'model']} ({df['fnr'].max():.2%})\n\n")
        
        f.write(f"### Latency Analysis\n")
        f.write(f"- Fastest: {fastest['model']} ({fastest['avg_latency']:.2f}s)\n")
        f.write(f"- Slowest: {df.loc[df['avg_latency'].idxmax(), 'model']} ({df['avg_latency'].max():.2f}s)\n")
        f.write(f"- Average: {df['avg_latency'].mean():.2f}s\n\n")
        
        if not repeat_df.empty:
            f.write("## Repeatability Analysis\n\n")
            f.write("### Score Consistency\n")
            variance_summary = repeat_df.groupby('model')['score_variance'].agg(['mean', 'std'])
            f.write(dataframe_to_markdown(variance_summary))
            f.write("\n\n")
            
            f.write("### Reason Consistency\n")
            consistency_summary = repeat_df.groupby('model')['reason_consistency'].agg(['mean', 'std'])
            f.write(dataframe_to_markdown(consistency_summary))
            f.write("\n\n")
        
        f.write("## Recommendations\n\n")
        
        if best_accuracy['fpr'] < 0.15:
            f.write(f"- **For Production Use**: {best_accuracy['model']} offers the best balance of accuracy with low false positive rate\n")
        
        if best_recall['fnr'] < 0.15:
            f.write(f"- **For High-Security Environments**: {best_recall['model']} minimizes false negatives (missed threats)\n")
        
        if fastest['avg_latency'] < 2.0:
            f.write(f"- **For Real-Time Applications**: {fastest['model']} provides the fastest inference suitable for browser-based detection\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("The following visualizations are available:\n")
        f.write("- `accuracy_comparison.png`: Overall accuracy, precision/recall, F1, and error rates\n")
        f.write("- `confusion_matrices.png`: Confusion matrices for all models\n")
        f.write("- `latency_comparison.png`: Inference speed comparison\n")
        f.write("- `repeatability_analysis.png`: Consistency and repeatability metrics\n")
        f.write("- `radar_chart.png`: Multi-metric radar comparison\n")
    
    print("✓ Completed ANALYSIS_REPORT.md")

