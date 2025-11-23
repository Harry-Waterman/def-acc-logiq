#!/usr/bin/env python3
"""Quick script to regenerate just the repeatability visualization."""
import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
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
    
    repeat_df = extract_repeatability_metrics(results)
    print(f"Extracted repeatability data: {len(repeat_df)} entries")
    
    print("Creating repeatability visualization...")
    plot_repeatability_analysis(repeat_df, str(output_dir))
    print("Done!")

