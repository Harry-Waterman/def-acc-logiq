#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of TinyRod benchmark results.
Generates visualizations, comparative analysis, and detailed reports.
"""

import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Color palette for consistency
COLORS = sns.color_palette("husl", 10)

def load_benchmark_results(results_dir: str = "../benchmark-results") -> List[Dict[str, Any]]:
    """Load all benchmark JSON files from the results directory."""
    results = []
    json_files = glob.glob(os.path.join(results_dir, "benchmark_results_*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract model name from filename
                filename = os.path.basename(file_path)
                model_name = filename.replace("benchmark_results_", "").replace("_seed42.json", "")
                data['model_name'] = model_name
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def extract_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract summary metrics from all benchmark results."""
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
            'fpr': accuracy.get('fpr', 0),  # False Positive Rate
            'fnr': accuracy.get('fnr', 0),  # False Negative Rate
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

def extract_repeatability_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Extract repeatability metrics from benchmark results."""
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

def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Create comparison charts for accuracy metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('accuracy', ascending=True)
    ax1.barh(df_sorted['model'], df_sorted['accuracy'], color=COLORS[0])
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Overall Accuracy')
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Precision vs Recall
    ax2 = axes[0, 1]
    for i, row in df.iterrows():
        ax2.scatter(row['recall'], row['precision'], s=200, alpha=0.7, 
                   label=row['model'] if i < len(df) else '')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # F1 Score
    ax3 = axes[1, 0]
    df_sorted_f1 = df.sort_values('f1', ascending=True)
    ax3.barh(df_sorted_f1['model'], df_sorted_f1['f1'], color=COLORS[2])
    ax3.set_xlabel('F1 Score')
    ax3.set_title('F1 Score Comparison')
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)
    
    # FPR vs FNR
    ax4 = axes[1, 1]
    for i, row in df.iterrows():
        ax4.scatter(row['fnr'], row['fpr'], s=200, alpha=0.7, 
                   label=row['model'] if i < len(df) else '')
    ax4.set_xlabel('False Negative Rate (FNR)')
    ax4.set_ylabel('False Positive Rate (FPR)')
    ax4.set_title('Error Rates: FPR vs FNR')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created accuracy_comparison.png")

def plot_confusion_matrix_comparison(df: pd.DataFrame, output_dir: str):
    """Create confusion matrix visualization for all models."""
    n_models = len(df)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrix Comparison', fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(df.iterrows()):
        cm = np.array([[row['tn'], row['fp']],
                       [row['fn'], row['tp']]])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   xticklabels=['Benign', 'Malicious'],
                   yticklabels=['Benign', 'Malicious'],
                   cbar_kws={'label': 'Count'})
        axes[idx].set_title(f"{row['model']}\nAccuracy: {row['accuracy']:.2%}")
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created confusion_matrices.png")

def plot_latency_comparison(df: pd.DataFrame, output_dir: str):
    """Create latency comparison chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_sorted = df.sort_values('avg_latency', ascending=True)
    bars = ax.barh(df_sorted['model'], df_sorted['avg_latency'], color=COLORS[4])
    
    ax.set_xlabel('Average Latency (seconds)')
    ax.set_title('Inference Latency Comparison')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row['avg_latency'] + 0.05, i, f"{row['avg_latency']:.2f}s", 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created latency_comparison.png")

def plot_repeatability_analysis(repeat_df: pd.DataFrame, output_dir: str):
    """Create repeatability analysis visualizations."""
    if repeat_df.empty:
        print("⚠ No repeatability data available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Repeatability Analysis', fontsize=16, fontweight='bold')
    
    # Check if all variance is zero (single run scenario)
    all_variance_zero = (repeat_df['score_variance'] == 0).all()
    
    # Score variance by model
    ax1 = axes[0, 0]
    variance_by_model = repeat_df.groupby('model')['score_variance'].mean().sort_values()
    
    # Set appropriate x-axis limits
    if all_variance_zero:
        # When all variance is 0, show a small range around 0 for visibility
        x_min, x_max = -0.05, 0.05
        ax1.set_xlim(x_min, x_max)
        # Add annotation explaining why
        ax1.text(0.5, 0.5, 'Note: All models show zero variance\n(only 1 run per email in this benchmark)',
                transform=ax1.transAxes, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
    else:
        # Auto-scale when there's actual variance
        x_min = min(0, variance_by_model.min() * 1.1)
        x_max = variance_by_model.max() * 1.1
        ax1.set_xlim(x_min, x_max)
    
    bars1 = ax1.barh(variance_by_model.index, variance_by_model.values, color=COLORS[5])
    # Add value labels on bars
    for i, (idx, val) in enumerate(variance_by_model.items()):
        label_x = val + (x_max - x_min) * 0.01 if val >= 0 else val - (x_max - x_min) * 0.01
        ax1.text(label_x, i, f'{val:.4f}', va='center', fontsize=9)
    
    ax1.set_xlabel('Average Score Variance')
    ax1.set_title('Consistency: Lower Variance = More Consistent')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Reason consistency by model
    ax2 = axes[0, 1]
    consistency_by_model = repeat_df.groupby('model')['reason_consistency'].mean().sort_values()
    
    bars2 = ax2.barh(consistency_by_model.index, consistency_by_model.values, color=COLORS[6])
    # Add value labels on bars
    for i, (idx, val) in enumerate(consistency_by_model.items()):
        ax2.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=9)
    
    ax2.set_xlabel('Average Reason Consistency')
    ax2.set_title('Reason Consistency: Higher = More Stable')
    ax2.set_xlim(0, 1.1)  # Slightly extend to show labels
    ax2.grid(axis='x', alpha=0.3)
    
    # Score variance distribution - use better visualization when all values are the same
    ax3 = axes[1, 0]
    if all_variance_zero:
        # When all variance is 0, show score_mean distribution instead
        for model in repeat_df['model'].unique():
            model_data = repeat_df[repeat_df['model'] == model]['score_mean']
            ax3.hist(model_data, alpha=0.5, label=model, bins=15, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Score Mean (0-100)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Distribution by Model\n(Note: Variance is 0 - showing score means instead)')
    else:
        # Normal variance distribution
        for model in repeat_df['model'].unique():
            model_data = repeat_df[repeat_df['model'] == model]['score_variance']
            ax3.hist(model_data, alpha=0.5, label=model, bins=20, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Score Variance')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Score Variance Distribution by Model')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.grid(alpha=0.3, axis='y')
    
    # Reason consistency distribution
    ax4 = axes[1, 1]
    all_consistency_one = (repeat_df['reason_consistency'] == 1.0).all()
    
    if all_consistency_one:
        # When all consistency is 1.0, show as a bar chart showing all models at 1.0
        models = repeat_df['model'].unique()
        ax4.barh(range(len(models)), [1.0] * len(models), color=COLORS[6], alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(models)))
        ax4.set_yticklabels(models)
        ax4.set_xlabel('Reason Consistency')
        ax4.set_xlim(0.9, 1.05)
        ax4.set_title('Reason Consistency by Model\n(All models: Perfect consistency = 1.0)')
        # Add value labels
        for i in range(len(models)):
            ax4.text(1.0, i, '1.00', va='center', ha='left', fontsize=9)
    else:
        # Normal distribution
        for model in repeat_df['model'].unique():
            model_data = repeat_df[repeat_df['model'] == model]['reason_consistency']
            ax4.hist(model_data, alpha=0.5, label=model, bins=20, edgecolor='black', linewidth=0.5)
        ax4.set_xlabel('Reason Consistency')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reason Consistency Distribution by Model')
        ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repeatability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created repeatability_analysis.png")

def plot_radar_chart(df: pd.DataFrame, output_dir: str):
    """Create radar chart comparing multiple metrics across models."""
    from math import pi
    
    # Normalize metrics to 0-1 scale for radar chart
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    normalized_df = df[['model'] + metrics_to_plot].copy()
    
    # Create radar chart for each model
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = [n / float(len(metrics_to_plot)) * 2 * pi for n in range(len(metrics_to_plot))]
    angles += angles[:1]  # Complete the circle
    
    for idx, row in df.iterrows():
        values = [row[m] for m in metrics_to_plot]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], alpha=0.7)
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Performance Comparison (Radar Chart)', 
                 size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created radar_chart.png")

def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table without requiring tabulate."""
    if df.empty:
        return ""
    
    # Get column names
    headers = list(df.columns)
    
    # Create header row
    markdown = "| " + " | ".join(str(h) for h in headers) + " |\n"
    markdown += "| " + " | ".join("---" for _ in headers) + " |\n"
    
    # Add data rows
    for _, row in df.iterrows():
        markdown += "| " + " | ".join(str(row[col]) for col in headers) + " |\n"
    
    return markdown

def generate_summary_report(df: pd.DataFrame, repeat_df: pd.DataFrame, output_dir: str):
    """Generate a comprehensive text report."""
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
        
        # Best accuracy
        best_accuracy = df.loc[df['accuracy'].idxmax()]
        f.write(f"### 🏆 Best Overall Accuracy: {best_accuracy['model']}\n")
        f.write(f"- Accuracy: {best_accuracy['accuracy']:.2%}\n")
        f.write(f"- Precision: {best_accuracy['precision']:.2%}\n")
        f.write(f"- Recall: {best_accuracy['recall']:.2%}\n")
        f.write(f"- F1 Score: {best_accuracy['f1']:.2%}\n\n")
        
        # Best precision
        best_precision = df.loc[df['precision'].idxmax()]
        f.write(f"### 🎯 Best Precision: {best_precision['model']}\n")
        f.write(f"- Precision: {best_precision['precision']:.2%} (fewest false positives)\n\n")
        
        # Best recall
        best_recall = df.loc[df['recall'].idxmax()]
        f.write(f"### 🔍 Best Recall: {best_recall['model']}\n")
        f.write(f"- Recall: {best_recall['recall']:.2%} (fewest false negatives)\n\n")
        
        # Best F1
        best_f1 = df.loc[df['f1'].idxmax()]
        f.write(f"### ⚖️ Best F1 Score: {best_f1['model']}\n")
        f.write(f"- F1: {best_f1['f1']:.2%} (balanced precision/recall)\n\n")
        
        # Fastest
        fastest = df.loc[df['avg_latency'].idxmin()]
        f.write(f"### ⚡ Fastest Inference: {fastest['model']}\n")
        f.write(f"- Average Latency: {fastest['avg_latency']:.2f} seconds\n\n")
        
        f.write("## Detailed Metrics Table\n\n")
        f.write(dataframe_to_markdown(df))
        f.write("\n\n")
        
        f.write("## Key Insights\n\n")
        
        # Accuracy range
        acc_range = df['accuracy'].max() - df['accuracy'].min()
        f.write(f"### Accuracy Variation\n")
        f.write(f"- Range: {df['accuracy'].min():.2%} to {df['accuracy'].max():.2%}\n")
        f.write(f"- Spread: {acc_range:.2%}\n")
        f.write(f"- Mean: {df['accuracy'].mean():.2%}\n\n")
        
        # Error rates analysis
        f.write(f"### Error Rate Analysis\n")
        f.write(f"- **False Positive Rate (FPR)**: Critical for user experience\n")
        f.write(f"  - Lowest: {df.loc[df['fpr'].idxmin(), 'model']} ({df['fpr'].min():.2%})\n")
        f.write(f"  - Highest: {df.loc[df['fpr'].idxmax(), 'model']} ({df['fpr'].max():.2%})\n")
        f.write(f"- **False Negative Rate (FNR)**: Critical for security\n")
        f.write(f"  - Lowest: {df.loc[df['fnr'].idxmin(), 'model']} ({df['fnr'].min():.2%})\n")
        f.write(f"  - Highest: {df.loc[df['fnr'].idxmax(), 'model']} ({df['fnr'].max():.2%})\n\n")
        
        # Latency analysis
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
        
        # Generate recommendations based on metrics
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
    
    print(f"✓ Created ANALYSIS_REPORT.md")

def main():
    """Main analysis pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark-results"
    output_dir = script_dir
    
    print("🔍 Loading benchmark results...")
    results = load_benchmark_results(str(results_dir))
    
    if not results:
        print("❌ No benchmark results found!")
        return
    
    print(f"✓ Loaded {len(results)} benchmark result files\n")
    
    # Extract metrics
    print("📊 Extracting metrics...")
    df = extract_summary_metrics(results)
    repeat_df = extract_repeatability_metrics(results)
    
    print(f"✓ Extracted metrics for {len(df)} models\n")
    
    # Generate visualizations
    print("📈 Creating visualizations...")
    plot_accuracy_comparison(df, str(output_dir))
    plot_confusion_matrix_comparison(df, str(output_dir))
    plot_latency_comparison(df, str(output_dir))
    if not repeat_df.empty:
        plot_repeatability_analysis(repeat_df, str(output_dir))
    plot_radar_chart(df, str(output_dir))
    
    print("\n📝 Generating summary report...")
    generate_summary_report(df, repeat_df, str(output_dir))
    
    # Save raw data as CSV
    df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    if not repeat_df.empty:
        repeat_df.to_csv(os.path.join(output_dir, 'repeatability_metrics.csv'), index=False)
    
    print("\n✅ Analysis complete! Check the benchmark-analysis folder for results.")
    print(f"\n📁 Output files:")
    print(f"   - ANALYSIS_REPORT.md")
    print(f"   - summary_metrics.csv")
    print(f"   - repeatability_metrics.csv")
    print(f"   - *.png (visualizations)")

if __name__ == "__main__":
    main()

