#!/usr/bin/env python3
"""Quick script to regenerate just the accuracy comparison visualization."""
import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 10
COLORS = sns.color_palette("husl", 10)

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

def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str):
    """Create comparison charts for accuracy metrics."""
    # Increase figure size to accommodate all models and legends
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # Accuracy
    ax1 = axes[0, 0]
    df_sorted = df.sort_values('accuracy', ascending=True)
    bars1 = ax1.barh(df_sorted['model'], df_sorted['accuracy'], color=COLORS[0])
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax1.text(row['accuracy'] + 0.01, i, f"{row['accuracy']:.2%}", 
                va='center', fontsize=9)
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Overall Accuracy')
    ax1.set_xlim(0, 1.1)  # Slightly extend to show labels
    ax1.grid(axis='x', alpha=0.3)
    
    # Precision vs Recall
    ax2 = axes[0, 1]
    # Use different colors for each model
    model_colors = COLORS[:len(df)]
    for i, (idx, row) in enumerate(df.iterrows()):
        ax2.scatter(row['recall'], row['precision'], s=200, alpha=0.7, 
                   color=model_colors[i], label=row['model'])
        # Add model name labels near points
        ax2.annotate(row['model'], (row['recall'], row['precision']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    # F1 Score
    ax3 = axes[1, 0]
    df_sorted_f1 = df.sort_values('f1', ascending=True)
    bars3 = ax3.barh(df_sorted_f1['model'], df_sorted_f1['f1'], color=COLORS[2])
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted_f1.iterrows()):
        ax3.text(row['f1'] + 0.01, i, f"{row['f1']:.2%}", 
                va='center', fontsize=9)
    ax3.set_xlabel('F1 Score')
    ax3.set_title('F1 Score Comparison')
    ax3.set_xlim(0, 1.1)  # Slightly extend to show labels
    ax3.grid(axis='x', alpha=0.3)
    
    # FPR vs FNR
    ax4 = axes[1, 1]
    for i, (idx, row) in enumerate(df.iterrows()):
        ax4.scatter(row['fnr'], row['fpr'], s=200, alpha=0.7, 
                   color=model_colors[i], label=row['model'])
        # Add model name labels near points
        ax4.annotate(row['model'], (row['fnr'], row['fpr']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax4.set_xlabel('False Negative Rate (FNR)')
    ax4.set_ylabel('False Positive Rate (FPR)')
    ax4.set_title('Error Rates: FPR vs FNR')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.legend(loc='upper right', fontsize=7, framealpha=0.9)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created accuracy_comparison.png")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark-results"
    output_dir = script_dir
    
    print("Loading benchmark results...")
    results = load_benchmark_results(str(results_dir))
    
    if not results:
        print("No benchmark results found!")
        exit(1)
    
    print(f"Loaded {len(results)} benchmark result files")
    
    df = extract_summary_metrics(results)
    print(f"Extracted metrics for {len(df)} models")
    
    print("Creating accuracy comparison visualization...")
    plot_accuracy_comparison(df, str(output_dir))
    print("Done!")

