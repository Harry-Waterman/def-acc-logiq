#!/usr/bin/env python3
"""
Comprehensive analysis and visualization of TinyRod benchmark results.
Generates visualizations, comparative analysis, and detailed reports.
"""

import json
import os
import glob
import warnings
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
COLORS = sns.color_palette("husl", 20)  # Extended palette for more models

def get_model_color_map(models: List[str]) -> Dict[str, tuple]:
    """
    Create a consistent color mapping for models that persists across all visualizations.
    Uses a deterministic approach based on sorted model names to ensure consistency.
    
    Args:
        models: List of model names (will be sorted internally)
    
    Returns:
        Dictionary mapping model names to RGB color tuples
    """
    # Sort models for consistent ordering (remove duplicates)
    sorted_models = sorted(set(models))
    
    # Use a color palette that's visually distinct
    # Set2 has 8 colors, so for more models, use husl which generates distinct colors
    if len(sorted_models) <= 8:
        palette = sns.color_palette("Set2", len(sorted_models))
    else:
        # Use husl palette for more models - it generates distinct colors for any number
        palette = sns.color_palette("husl", len(sorted_models))
    
    return dict(zip(sorted_models, palette))

# Visualization standards and constants
# Score range (0-100)
SCORE_MIN = 0
SCORE_MAX = 100
SCORE_BIN_WIDTH = 5
SCORE_BIN_OFFSET = SCORE_BIN_WIDTH / 2  # Half bin width for centering

# Consistency range (0-1)
CONSISTENCY_MIN = 0
CONSISTENCY_MAX = 1.0
CONSISTENCY_BIN_WIDTH = 0.05
CONSISTENCY_BIN_OFFSET = CONSISTENCY_BIN_WIDTH / 2

# Duration bin settings
DURATION_BIN_WIDTH = 2  # 2-second bins
DURATION_BIN_OFFSET = DURATION_BIN_WIDTH / 2

# Variance bin settings (dynamic based on range)
VARIANCE_SMALL_BIN_WIDTH = 0.5  # For variance <= 10
VARIANCE_LARGE_BIN_WIDTH = 1.0  # For variance > 10
VARIANCE_SMALL_BIN_OFFSET = VARIANCE_SMALL_BIN_WIDTH / 2
VARIANCE_LARGE_BIN_OFFSET = VARIANCE_LARGE_BIN_WIDTH / 2

def create_aligned_bins(min_val: float, max_val: float, bin_width: float, bin_offset: float = None) -> np.ndarray:
    """
    Create bins aligned with axis values for proper centering.
    
    Args:
        min_val: Minimum value to include
        max_val: Maximum value to include
        bin_width: Width of each bin
        bin_offset: Half bin width for centering (defaults to bin_width/2)
    
    Returns:
        Array of bin edges that will center bars at min_val, min_val+bin_width, ..., max_val
    """
    if bin_offset is None:
        bin_offset = bin_width / 2
    
    # Start slightly before min to center first bin at min_val
    start = min_val - bin_offset
    # End slightly after max to include max_val in last bin (arange is exclusive)
    end = max_val + bin_width + bin_offset
    
    return np.arange(start, end, bin_width)

def get_score_bins() -> np.ndarray:
    """Get bins for score distributions (0-100)."""
    return create_aligned_bins(SCORE_MIN, SCORE_MAX, SCORE_BIN_WIDTH, SCORE_BIN_OFFSET)

def get_consistency_bins() -> np.ndarray:
    """Get bins for consistency distributions (0-1)."""
    return create_aligned_bins(CONSISTENCY_MIN, CONSISTENCY_MAX, CONSISTENCY_BIN_WIDTH, CONSISTENCY_BIN_OFFSET)

def get_duration_bins(max_duration: float) -> np.ndarray:
    """Get bins for duration distributions."""
    # Round up to nearest bin width
    max_bin = int(np.ceil(max_duration / DURATION_BIN_WIDTH) * DURATION_BIN_WIDTH) + DURATION_BIN_WIDTH
    return create_aligned_bins(0, max_bin, DURATION_BIN_WIDTH, DURATION_BIN_OFFSET)

def get_variance_bins(max_variance: float) -> np.ndarray:
    """Get bins for variance distributions."""
    if max_variance <= 10:
        bin_width = VARIANCE_SMALL_BIN_WIDTH
        bin_offset = VARIANCE_SMALL_BIN_OFFSET
    else:
        bin_width = VARIANCE_LARGE_BIN_WIDTH
        bin_offset = VARIANCE_LARGE_BIN_OFFSET
    
    if max_variance > 0:
        max_bin = int(np.ceil(max_variance / bin_width) * bin_width) + bin_width
        return create_aligned_bins(0, max_bin, bin_width, bin_offset)
    else:
        # Default bins for zero variance case
        return create_aligned_bins(0, 1.0, 0.1, 0.05)

def get_score_axis_limits() -> tuple:
    """Get x-axis limits for score plots."""
    bins = get_score_bins()
    return (bins[0], bins[-1] - SCORE_BIN_WIDTH)  # Show actual 0-100 range

def get_consistency_axis_limits() -> tuple:
    """Get x-axis limits for consistency plots."""
    bins = get_consistency_bins()
    return (bins[0], bins[-1] - CONSISTENCY_BIN_WIDTH)  # Show actual 0-1 range

def load_benchmark_results(results_dir: str = "../benchmark-results", include_repeatability: bool = False) -> List[Dict[str, Any]]:
    """
    Load benchmark JSON files from the results directory.
    
    Args:
        results_dir: Directory containing benchmark result files
        include_repeatability: If True, includes files with '_repeatability_' in name.
                              If False, excludes files with '_repeatability_' in name.
    
    Returns:
        List of benchmark result dictionaries with 'model_name' added
    """
    results = []
    json_files = glob.glob(os.path.join(results_dir, "benchmark_results_*.json"))
    
    for file_path in json_files:
        filename = os.path.basename(file_path)
        
        # Filter based on repeatability flag
        is_repeatability_file = '_repeatability_' in filename
        if is_repeatability_file and not include_repeatability:
            continue  # Skip repeatability files when loading accuracy results
        if not is_repeatability_file and include_repeatability:
            continue  # Skip non-repeatability files when loading repeatability results
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # Extract model name from filename
                # Pattern: benchmark_results_MODEL[-repeatability]_seed-info.json
                # This preserves -reasoning suffix and other distinctions that may not be in JSON config
                import re
                # Match: benchmark_results_<model_name>_seed or benchmark_results_<model_name>_repeatability_seed
                match = re.search(r'benchmark_results_(.+?)(?:_repeatability)?_seed', filename)
                if match:
                    model_name = match.group(1)
                else:
                    # Fallback: extract everything between benchmark_results_ and .json
                    model_name = filename.replace("benchmark_results_", "").replace(".json", "")
                    # Remove _repeatability if present
                    model_name = model_name.replace("_repeatability", "")
                    # Remove any trailing _seed pattern if present
                    model_name = re.sub(r'_seed.*$', '', model_name)
                
                data['model_name'] = model_name
                data['is_repeatability_file'] = is_repeatability_file
                results.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return results

def extract_summary_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract summary metrics from accuracy benchmark results.
    Excludes repeatability-only files (they don't have accuracy_results).
    """
    metrics = []
    
    for result in results:
        # Skip repeatability-only files (they don't have accuracy_results)
        if result.get('is_repeatability_file', False):
            continue
            
        config = result.get('config', {})
        accuracy = result.get('accuracy_results', {})
        model_name = result.get('model_name', 'unknown')
        model_full = config.get('model', 'unknown')
        
        metrics.append({
            'model': model_name,
            'model_full': model_full,
            'model_size': extract_model_size(model_full),  # Extract model size in billions
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
    """
    Extract repeatability metrics from benchmark results.
    Only processes results from files with '_repeatability_' in the filename.
    """
    repeatability_data = []
    
    for result in results:
        # Only process repeatability files
        if not result.get('is_repeatability_file', False):
            continue
            
        model_name = result.get('model_name', 'unknown')
        config = result.get('config', {})
        repeatability = result.get('repeatability_results', [])
        
        for item in repeatability:
            repeatability_data.append({
                'model': model_name,
                'ground_truth': item.get('ground_truth', 0),
                'score_mean': item.get('score_mean', 0),
                'score_variance': item.get('score_variance', 0),
                'reason_consistency': item.get('reason_consistency', 0),
                'temperature': config.get('temperature', 0),
            })
    
    return pd.DataFrame(repeatability_data)

def extract_per_email_metrics(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Extract per-email accuracy statistics from detailed_results.
    Excludes repeatability-only files (they don't have accuracy_results with detailed_results).
    """
    per_email_data = []
    
    for result in results:
        # Skip repeatability-only files (they don't have accuracy_results with detailed_results)
        if result.get('is_repeatability_file', False):
            continue
            
        model_name = result.get('model_name', 'unknown')
        accuracy = result.get('accuracy_results', {})
        detailed = accuracy.get('detailed_results', [])
        
        for item in detailed:
            # Handle ERROR/timeout cases - they don't have a status but should be tracked
            status = item.get('status', '')
            if item.get('prediction') == 'ERROR' or 'error' in item:
                # Categorize different types of errors
                error_msg = item.get('error', '').lower()
                if 'timeout' in error_msg or item.get('duration', 0) >= 60:
                    status = 'TIMEOUT'
                elif 'parse' in error_msg and 'json' in error_msg:
                    status = 'JSON_ERROR'
                else:
                    status = 'ERROR'
            
            per_email_data.append({
                'model': model_name,
                'index': item.get('index', 0),
                'ground_truth': item.get('ground_truth', 0),
                'prediction': item.get('prediction', 0),
                'score': item.get('score', 0),
                'correct': item.get('correct', False),
                'status': status,
                'duration': item.get('duration', 0),
                'has_error': item.get('prediction') == 'ERROR' or 'error' in item,
            })
    
    return pd.DataFrame(per_email_data)

def extract_model_size(model_name: str) -> float:
    """Extract model size in billions from model name (e.g., 'qwen3-4b' -> 4.0)."""
    import re
    # Try to find pattern like "4b", "1.2b", "8b", etc.
    match = re.search(r'(\d+\.?\d*)[bB]', model_name)
    if match:
        return float(match.group(1))
    # Try to find pattern like "-4-" or "-1.2-"
    match = re.search(r'-(\d+\.?\d*)-', model_name)
    if match:
        return float(match.group(1))
    return 0.0

def calculate_bestness_score(df: pd.DataFrame) -> pd.Series:
    """
    Calculate a composite 'bestness' score for each model.
    Higher score = better overall performance.
    
    Combines normalized metrics:
    - Higher is better: accuracy, precision, recall, F1, success_rate
    - Lower is better: latency, FPR, FNR, error rates (inverted)
    """
    scores = pd.Series(index=df.index, dtype=float)
    
    # Metrics where higher is better (normalize to 0-1)
    higher_better = ['accuracy', 'precision', 'recall', 'f1']
    if 'success_rate' in df.columns:
        higher_better.append('success_rate')
    
    # Metrics where lower is better (invert after normalizing)
    lower_better = ['avg_latency', 'fpr', 'fnr']
    if 'total_error_rate' in df.columns:
        lower_better.extend(['total_error_rate', 'timeout_rate', 'json_error_rate'])
    
    for idx in df.index:
        score = 0.0
        count = 0
        
        # Add higher-is-better metrics
        for metric in higher_better:
            if metric in df.columns and pd.notna(df.loc[idx, metric]):
                # Normalize to 0-1 (assuming values are already 0-1 for most metrics)
                val = df.loc[idx, metric]
                score += val
                count += 1
        
        # Add lower-is-better metrics (inverted)
        for metric in lower_better:
            if metric in df.columns and pd.notna(df.loc[idx, metric]):
                val = df.loc[idx, metric]
                # Normalize and invert: (max - val) / (max - min)
                col_max = df[metric].max()
                col_min = df[metric].min()
                if col_max > col_min:
                    normalized = (col_max - val) / (col_max - col_min)
                    score += normalized
                    count += 1
        
        # Average score
        scores.loc[idx] = score / count if count > 0 else 0.0
    
    return scores

def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create comparison charts for accuracy metrics."""
    # Calculate bestness score and sort by it (best at top)
    df_with_score = df.copy()
    df_with_score['_bestness'] = calculate_bestness_score(df_with_score)
    df_sorted_by_bestness = df_with_score.sort_values('_bestness', ascending=False)
    
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    # Increase figure size to accommodate all models and legends
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Model Performance Comparison\n(Models ordered by overall bestness score)', fontsize=16, fontweight='bold')
    
    # Accuracy - sorted by bestness (best at top)
    ax1 = axes[0, 0]
    df_sorted = df_sorted_by_bestness.sort_values('accuracy', ascending=True)
    
    # Ensure accuracy values are in 0-1 range (normalize if they're percentages)
    accuracy_values = df_sorted['accuracy'].copy()
    if accuracy_values.max() > 1:
        accuracy_values = accuracy_values / 100
    
    # Use model-specific colors for each bar
    bar_colors = [model_colors.get(model, COLORS[0]) for model in df_sorted['model']]
    bars1 = ax1.barh(df_sorted['model'], accuracy_values, color=bar_colors)
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        acc_val = accuracy_values.iloc[i]
        ax1.text(acc_val + 0.01, i, f"{acc_val:.2%}", 
                va='center', fontsize=9)
    ax1.set_xlabel('Accuracy')
    ax1.set_title('Overall Accuracy')
    ax1.set_xlim(0, max(1.1, accuracy_values.max() * 1.1))  # Ensure 0 is at left, extend for labels
    ax1.grid(axis='x', alpha=0.3)
    
    # Precision vs Recall
    ax2 = axes[0, 1]
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    # Normalize precision and recall to 0-1 range if needed
    precision_values = df['precision'].copy()
    recall_values = df['recall'].copy()
    if precision_values.max() > 1:
        precision_values = precision_values / 100
    if recall_values.max() > 1:
        recall_values = recall_values / 100
    
    for idx, row in df.iterrows():
        model_color = model_colors.get(row['model'], COLORS[0])
        prec_val = precision_values.loc[idx]
        recall_val = recall_values.loc[idx]
        ax2.scatter(recall_val, prec_val, s=200, alpha=0.7, 
                   color=model_color, label=row['model'])
        # Add model name labels near points
        ax2.annotate(row['model'], (recall_val, prec_val),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision vs Recall')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='lower left', fontsize=7, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    # F1 Score - sorted by bestness (best at top)
    ax3 = axes[1, 0]
    df_sorted_f1 = df_sorted_by_bestness.sort_values('f1', ascending=True)
    
    # Ensure F1 values are in 0-1 range (normalize if they're percentages)
    f1_values = df_sorted_f1['f1'].copy()
    if f1_values.max() > 1:
        f1_values = f1_values / 100
    
    # Use model-specific colors for each bar
    bar_colors_f1 = [model_colors.get(model, COLORS[0]) for model in df_sorted_f1['model']]
    bars3 = ax3.barh(df_sorted_f1['model'], f1_values, color=bar_colors_f1)
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted_f1.iterrows()):
        f1_val = f1_values.iloc[i]
        ax3.text(f1_val + 0.01, i, f"{f1_val:.2%}", 
                va='center', fontsize=9)
    ax3.set_xlabel('F1 Score')
    ax3.set_title('F1 Score Comparison')
    ax3.set_xlim(0, max(1.1, f1_values.max() * 1.1))  # Ensure 0 is at left, extend for labels
    ax3.grid(axis='x', alpha=0.3)
    
    # FPR vs FNR
    ax4 = axes[1, 1]
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    # Normalize FPR and FNR to 0-1 range if needed
    fpr_values = df['fpr'].copy()
    fnr_values = df['fnr'].copy()
    if fpr_values.max() > 1:
        fpr_values = fpr_values / 100
    if fnr_values.max() > 1:
        fnr_values = fnr_values / 100
    
    for idx, row in df.iterrows():
        model_color = model_colors.get(row['model'], COLORS[0])
        fpr_val = fpr_values.loc[idx]
        fnr_val = fnr_values.loc[idx]
        ax4.scatter(fnr_val, fpr_val, s=200, alpha=0.7, 
                   color=model_color, label=row['model'])
        # Add model name labels near points, adjust position to avoid legend
        ax4.annotate(row['model'], (fnr_val, fpr_val),
                    xytext=(5, 5), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='gray'))
    ax4.set_xlabel('False Negative Rate (FNR)')
    ax4.set_ylabel('False Positive Rate (FPR)')
    ax4.set_title('Error Rates: FPR vs FNR')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    # Move legend to avoid overlapping with annotations
    ax4.legend(loc='lower left', fontsize=7, framealpha=0.9, ncol=2, columnspacing=0.5)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created accuracy_comparison.png")

def plot_confusion_matrix_comparison(df: pd.DataFrame, output_dir: str):
    """Create confusion matrix visualization for all models."""
    # Sort by bestness score (best first)
    df_with_score = df.copy()
    df_with_score['_bestness'] = calculate_bestness_score(df_with_score)
    df_sorted = df_with_score.sort_values('_bestness', ascending=False)
    
    n_models = len(df_sorted)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    fig.suptitle('Confusion Matrix Comparison\n(Models ordered by overall bestness score)', fontsize=16, fontweight='bold')
    
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
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

def plot_latency_comparison(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create latency comparison chart."""
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    if df.empty:
        print("⚠ No data for latency comparison")
        return
    
    # Ensure we have unique models (in case of duplicates)
    df_unique = df.drop_duplicates(subset=['model'], keep='first').copy()
    
    print(f"  Plotting latency for {len(df_unique)} models: {sorted(df_unique['model'].unique())}")
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(df_unique) * 0.6)))
    
    # Sort by latency value (fastest first) - handle outliers by using median or excluding extreme values
    # Check for outliers (models with very high latency due to timeouts)
    latency_median = df_unique['avg_latency'].median()
    latency_q75 = df_unique['avg_latency'].quantile(0.75)
    iqr = df_unique['avg_latency'].quantile(0.75) - df_unique['avg_latency'].quantile(0.25)
    outlier_threshold = latency_q75 + 1.5 * iqr if iqr > 0 else df_unique['avg_latency'].max()
    
    # Sort by latency (ascending - fastest/lowest at top of chart)
    # For barh charts, ascending order puts fastest (lowest values) at top
    df_sorted = df_unique.sort_values('avg_latency', ascending=True).reset_index(drop=True)
    
    # Use model-specific colors for each bar
    bar_colors = [model_colors.get(model, COLORS[0]) for model in df_sorted['model']]
    bars = ax.barh(df_sorted['model'], df_sorted['avg_latency'], color=bar_colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Average Latency (seconds)')
    ax.set_title('Inference Latency Comparison\n(Sorted by latency: fastest first)')
    ax.grid(axis='x', alpha=0.3)
    
    # Set x-axis limit to accommodate all bars, including outliers
    max_latency = df_sorted['avg_latency'].max()
    ax.set_xlim(0, max_latency * 1.1)
    
    # Handle outliers - if any model has extremely high latency, add a note
    outlier_models = df_sorted[df_sorted['avg_latency'] > outlier_threshold]
    if len(outlier_models) > 0:
        ax.text(0.98, 0.02, f'Note: {len(outlier_models)} model(s) have outlier latency', 
                transform=ax.transAxes, fontsize=8, ha='right',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        # For very high latency, show value but maybe truncate display
        latency_val = row['avg_latency']
        if latency_val > outlier_threshold:
            label = f"{latency_val:.1f}s*"
        else:
            label = f"{latency_val:.2f}s"
        # Position label at the end of the bar
        label_x = latency_val + max(0.05, max_latency * 0.02)
        ax.text(label_x, i, label, 
               va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created latency_comparison.png ({len(df_sorted)} models)")

def plot_repeatability_analysis(repeat_df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create repeatability analysis visualizations."""
    if repeat_df.empty:
        print("⚠ No repeatability data available")
        return
    
    if model_colors is None:
        model_colors = get_model_color_map(repeat_df['model'].unique())
    
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
    
    # Use model-specific colors for each bar
    bar_colors_variance = [model_colors.get(model, COLORS[0]) for model in variance_by_model.index]
    bars1 = ax1.barh(variance_by_model.index, variance_by_model.values, color=bar_colors_variance)
    # Add value labels on bars
    for i, (idx, val) in enumerate(variance_by_model.items()):
        label_x = val + (x_max - x_min) * 0.01 if val >= 0 else val - (x_max - x_min) * 0.01
        ax1.text(label_x, i, f'{val:.4f}', va='center', fontsize=9)
    
    ax1.set_xlabel('Average Score Variance')
    ax1.set_title('Consistency: Lower Variance = More Consistent')
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # Reason consistency by model - sort descending (higher consistency = better = higher in chart)
    ax2 = axes[0, 1]
    consistency_by_model = repeat_df.groupby('model')['reason_consistency'].mean().sort_values(ascending=False)
    
    # Use model-specific colors for each bar
    bar_colors_consistency = [model_colors.get(model, COLORS[0]) for model in consistency_by_model.index]
    bars2 = ax2.barh(consistency_by_model.index, consistency_by_model.values, color=bar_colors_consistency)
    # Add value labels on bars
    for i, (idx, val) in enumerate(consistency_by_model.items()):
        ax2.text(val + 0.02, i, f'{val:.2f}', va='center', fontsize=9)
    
    ax2.set_xlabel('Average Reason Consistency')
    ax2.set_title('Reason Consistency: Same reasons across runs\n(Not accuracy - measures repeatability)')
    ax2.set_xlim(0, 1.1)  # Slightly extend to show labels
    ax2.grid(axis='x', alpha=0.3)
    
    # Score Mean Distribution by Email Type (Benign vs Malicious)
    ax3 = axes[1, 0]
    repeat_df_copy = repeat_df.copy()
    repeat_df_copy['email_type'] = repeat_df_copy['ground_truth'].map({0: 'Benign', 1: 'Malicious'})
    
    # Create bins aligned with score range (0-100)
    score_mean_bins = get_score_bins()
    
    # Plot histogram for each model, grouped by email type
    models = repeat_df['model'].unique()
    x = np.arange(len(models))
    width = 0.35
    
    # Calculate mean scores by model and email type
    benign_means = []
    malicious_means = []
    for model in models:
        model_data = repeat_df_copy[repeat_df_copy['model'] == model]
        benign_mean = model_data[model_data['email_type'] == 'Benign']['score_mean'].mean() if len(model_data[model_data['email_type'] == 'Benign']) > 0 else 0
        malicious_mean = model_data[model_data['email_type'] == 'Malicious']['score_mean'].mean() if len(model_data[model_data['email_type'] == 'Malicious']) > 0 else 0
        benign_means.append(benign_mean)
        malicious_means.append(malicious_mean)
    
    # Create grouped bar chart
    for i, model in enumerate(models):
        model_color = model_colors.get(model, COLORS[0])
        ax3.bar(x[i] - width/2, benign_means[i], width, label='Benign' if i == 0 else '', 
               color=model_color, alpha=0.6, edgecolor='black')
        ax3.bar(x[i] + width/2, malicious_means[i], width, label='Malicious' if i == 0 else '', 
               color=model_color, alpha=0.9, edgecolor='black')
    
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Average Score Mean')
    ax3.set_title('Score Mean by Model and Email Type')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend(title='Email Type', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    
    # Reason consistency distribution
    ax4 = axes[1, 1]
    all_consistency_one = (repeat_df['reason_consistency'] == 1.0).all()
    
    if all_consistency_one:
        # When all consistency is 1.0, show as a bar chart showing all models at 1.0
        models = repeat_df['model'].unique()
        bar_colors = [model_colors.get(model, COLORS[0]) for model in models]
        ax4.barh(range(len(models)), [1.0] * len(models), color=bar_colors, alpha=0.7, edgecolor='black')
        ax4.set_yticks(range(len(models)))
        ax4.set_yticklabels(models)
        ax4.set_xlabel('Reason Consistency')
        ax4.set_xlim(0.9, 1.05)
        ax4.set_title('Reason Consistency by Model\n(1.0 = Same reasons across runs, not perfect accuracy)')
        # Add value labels
        for i in range(len(models)):
            ax4.text(1.0, i, '1.00', va='center', ha='left', fontsize=9)
        # Add annotation explaining the distinction
        ax4.text(0.975, len(models) - 0.5, 'Note: Consistency measures\nrepeatability, not correctness',
                ha='right', va='center', fontsize=8, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # Normal distribution
        # Create bins aligned with consistency range (0-1)
        consistency_bins = get_consistency_bins()
        for model in repeat_df['model'].unique():
            model_data = repeat_df[repeat_df['model'] == model]['reason_consistency']
            model_color = model_colors.get(model, COLORS[0])
            ax4.hist(model_data, alpha=0.5, label=model, bins=consistency_bins, edgecolor='black', linewidth=0.5, align='mid', color=model_color)
        ax4.set_xlim(*get_consistency_axis_limits())  # Set explicit limits to show 0-1 range
        ax4.set_xlabel('Reason Consistency')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Reason Consistency Distribution by Model')
        ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'repeatability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created repeatability_analysis.png")

def plot_radar_chart(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create radar chart comparing multiple metrics across models."""
    from math import pi
    
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
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
        model_color = model_colors.get(row['model'], COLORS[0])
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], alpha=0.7, color=model_color)
        ax.fill(angles, values, alpha=0.15, color=model_color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics_to_plot])
    ax.set_ylim(0, 1)
    ax.set_title('Multi-Metric Performance Comparison (Radar Chart)', 
                 size=14, fontweight='bold', pad=20)
    # Move legend further outside to avoid overlap
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0), fontsize=9, framealpha=0.9)
    ax.grid(True)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on right for legend
    plt.savefig(os.path.join(output_dir, 'radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created radar_chart.png")

def plot_latency_accuracy_tradeoff(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create latency vs accuracy tradeoff scatter plot."""
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(13, 9))  # Increased size for better spacing
    
    # Plot each model - use opacity to indicate error rate if available
    for idx, row in df.iterrows():
        model_color = model_colors.get(row['model'], COLORS[0])
        # Adjust opacity and edge width based on error rate
        if 'total_error_rate' in df.columns:
            error_rate = row.get('total_error_rate', 0)
            alpha = 0.3 if error_rate > 0.1 else 0.7
            edge_width = 3 if error_rate > 0.1 else 1.5
        else:
            alpha = 0.7
            edge_width = 1.5
        
        ax.scatter(row['avg_latency'], row['accuracy'], s=300, alpha=alpha, 
                  color=model_color, label=row['model'], edgecolors='black', linewidth=edge_width)
        # Add model name labels with smart positioning to avoid overlaps
        # Position labels based on quadrant
        if row['avg_latency'] < df['avg_latency'].median() and row['accuracy'] > df['accuracy'].median():
            # Top-left: label to the right
            xytext_offset = (10, 0)
        elif row['avg_latency'] >= df['avg_latency'].median() and row['accuracy'] > df['accuracy'].median():
            # Top-right: label to the left
            xytext_offset = (-10, 0)
        elif row['avg_latency'] < df['avg_latency'].median() and row['accuracy'] <= df['accuracy'].median():
            # Bottom-left: label to the right
            xytext_offset = (10, -12)
        else:
            # Bottom-right: label to the left
            xytext_offset = (-10, -12)
        
        ax.annotate(row['model'], (row['avg_latency'], row['accuracy']),
                   xytext=xytext_offset, textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add quadrant lines
    median_latency = df['avg_latency'].median()
    median_accuracy = df['accuracy'].median()
    ax.axvline(x=median_latency, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=median_accuracy, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add quadrant labels
    ax.text(0.02, 0.98, 'High Accuracy\nLow Latency\n(Ideal)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    ax.text(0.98, 0.98, 'High Accuracy\nHigh Latency', 
           transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    ax.text(0.02, 0.02, 'Low Accuracy\nLow Latency', 
           transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    ax.text(0.98, 0.02, 'Low Accuracy\nHigh Latency\n(Worst)', 
           transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    ax.set_xlabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    title = 'Latency vs Accuracy Tradeoff'
    if 'total_error_rate' in df.columns:
        title += '\n(Models with high error rates shown with reduced opacity)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    # Move legend to avoid overlapping with quadrant labels
    ax.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), fontsize=8, framealpha=0.9, 
             ncol=1, frameon=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_accuracy_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created latency_accuracy_tradeoff.png")

def plot_quadrant_analysis(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create quadrant analysis showing accuracy vs latency with clear quadrants."""
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(14, 9))  # Increased size for better spacing
    
    # Calculate thresholds (median or mean)
    latency_threshold = df['avg_latency'].median()
    accuracy_threshold = df['accuracy'].median()
    
    # Categorize models into quadrants
    quadrants = {
        'High Acc / Low Lat': [],
        'High Acc / High Lat': [],
        'Low Acc / Low Lat': [],
        'Low Acc / High Lat': []
    }
    
    for idx, row in df.iterrows():
        if row['accuracy'] >= accuracy_threshold and row['avg_latency'] <= latency_threshold:
            quadrants['High Acc / Low Lat'].append((row['model'], row['avg_latency'], row['accuracy']))
        elif row['accuracy'] >= accuracy_threshold and row['avg_latency'] > latency_threshold:
            quadrants['High Acc / High Lat'].append((row['model'], row['avg_latency'], row['accuracy']))
        elif row['accuracy'] < accuracy_threshold and row['avg_latency'] <= latency_threshold:
            quadrants['Low Acc / Low Lat'].append((row['model'], row['avg_latency'], row['accuracy']))
        else:
            quadrants['Low Acc / High Lat'].append((row['model'], row['avg_latency'], row['accuracy']))
    
    # Plot quadrant boundaries
    ax.axvline(x=latency_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Latency Threshold: {latency_threshold:.2f}s')
    ax.axhline(y=accuracy_threshold, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Accuracy Threshold: {accuracy_threshold:.2%}')
    
    # Plot models with different colors for each quadrant
    quadrant_colors = {
        'High Acc / Low Lat': 'green',
        'High Acc / High Lat': 'orange',
        'Low Acc / Low Lat': 'blue',
        'Low Acc / High Lat': 'red'
    }
    
    for quadrant, models in quadrants.items():
        for idx, (model_name, latency, accuracy) in enumerate(models):
            # Use consistent model color instead of quadrant color
            model_color = model_colors.get(model_name, quadrant_colors[quadrant])
            ax.scatter(latency, accuracy, s=400, alpha=0.7, 
                      color=model_color, label=quadrant if idx == 0 else '',
                      edgecolors='black', linewidth=2, zorder=5)
            # Adjust annotation position to avoid legend area (upper left) and other labels
            # Use different offsets based on quadrant position
            if latency < latency_threshold and accuracy > accuracy_threshold:
                # Top-left quadrant - place label to the right, below legend
                xytext_offset = (20, -10)
            elif latency >= latency_threshold and accuracy > accuracy_threshold:
                # Top-right quadrant - place label to the left
                xytext_offset = (-20, 5)
            elif latency < latency_threshold and accuracy <= accuracy_threshold:
                # Bottom-left quadrant - place label to the right
                xytext_offset = (20, -15)
            else:
                # Bottom-right quadrant - place label to the left
                xytext_offset = (-20, -15)
            
            ax.annotate(model_name, (latency, accuracy),
                       xytext=xytext_offset, textcoords='offset points', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'))
    
    # Fill quadrants with semi-transparent colors
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.fill_between([xlim[0], latency_threshold], accuracy_threshold, ylim[1], 
                    alpha=0.1, color='green', label='Best Quadrant')
    ax.fill_between([latency_threshold, xlim[1]], accuracy_threshold, ylim[1], 
                    alpha=0.1, color='orange')
    ax.fill_between([xlim[0], latency_threshold], ylim[0], accuracy_threshold, 
                    alpha=0.1, color='blue')
    ax.fill_between([latency_threshold, xlim[1]], ylim[0], accuracy_threshold, 
                    alpha=0.1, color='red', label='Worst Quadrant')
    
    ax.set_xlabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Quadrant Analysis: Accuracy vs Latency', fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3, zorder=0)
    # Move legend to lower right to avoid overlapping with model labels in upper left
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, 
             ncol=1, columnspacing=1, handletextpad=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quadrant_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created quadrant_analysis.png")

def plot_metrics_heatmap(df: pd.DataFrame, output_dir: str):
    """Create comprehensive metrics heatmap."""
    # Select metrics to display - include error rates if available
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'avg_latency', 'fpr', 'fnr']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Latency (s)', 'FPR', 'FNR']
    
    # Add error rate metrics if available
    if 'total_error_rate' in df.columns:
        metrics.extend(['total_error_rate', 'timeout_rate', 'json_error_rate'])
        metric_labels.extend(['Total Error Rate', 'Timeout Rate', 'JSON Error Rate'])
    
    # Calculate bestness score and sort models (best at top)
    df_with_score = df.copy()
    df_with_score['_bestness'] = calculate_bestness_score(df_with_score)
    df_sorted = df_with_score.sort_values('_bestness', ascending=False)
    
    # Create matrix for heatmap (sorted by bestness)
    heatmap_data = df_sorted[['model'] + metrics].set_index('model')
    heatmap_data.columns = metric_labels
    
    # Normalize all metrics to 0-1 scale for consistent heatmap
    # For latency, FPR, FNR: lower is better, so we invert them (lowest = highest score)
    # For all other metrics: higher is better
    heatmap_data_normalized = heatmap_data.copy()
    
    # Metrics where lower is better (need inversion)
    lower_is_better = ['Latency (s)', 'FPR', 'FNR', 'Total Error Rate', 'Timeout Rate', 'JSON Error Rate']
    
    for col in heatmap_data_normalized.columns:
        min_val = heatmap_data_normalized[col].min()
        max_val = heatmap_data_normalized[col].max()
        if max_val > min_val:
            if col in lower_is_better:
                # Invert: lower value should map to higher normalized score
                # Formula: (max - value) / (max - min) ensures lowest value = 1.0, highest = 0.0
                heatmap_data_normalized[col] = (max_val - heatmap_data_normalized[col]) / (max_val - min_val)
            else:
                # Normalize normally: higher value = higher score
                heatmap_data_normalized[col] = (heatmap_data_normalized[col] - min_val) / (max_val - min_val)
    
    # Create figure with single heatmap showing raw values
    fig, ax = plt.subplots(1, 1, figsize=(12, max(6, len(df) * 0.8)))
    
    # For raw values, we need to handle latency, FPR, FNR differently (lower is better)
    # We'll create a normalized version for coloring, but show raw values in annotations
    heatmap_data_raw = heatmap_data.copy()
    heatmap_data_raw_for_color = heatmap_data.copy()
    
    # Normalize for color mapping (invert latency, FPR, FNR)
    for col in heatmap_data_raw_for_color.columns:
        min_val = heatmap_data_raw_for_color[col].min()
        max_val = heatmap_data_raw_for_color[col].max()
        if max_val > min_val:
            if col in lower_is_better:
                # Invert for color: lower value = better color
                heatmap_data_raw_for_color[col] = (max_val - heatmap_data_raw_for_color[col]) / (max_val - min_val)
            else:
                # Normalize normally
                heatmap_data_raw_for_color[col] = (heatmap_data_raw_for_color[col] - min_val) / (max_val - min_val)
    
    # Create heatmap with raw values as annotations but normalized values for color
    sns.heatmap(heatmap_data_raw_for_color, annot=heatmap_data_raw, cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Performance (normalized for color, raw values shown)'}, fmt='.3f', 
                annot_kws={'size': 9})
    ax.set_title('Comprehensive Metrics Heatmap\n(Actual values shown, color indicates performance)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('Metrics', fontsize=10)
    ax.set_ylabel('Models', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created metrics_heatmap.png")

def plot_pareto_frontier(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create Pareto frontier showing efficiency frontier for accuracy vs latency."""
    if model_colors is None:
        model_colors = get_model_color_map(df['model'].unique())
    
    fig, ax = plt.subplots(figsize=(13, 9))  # Increased size for better spacing
    
    # For Pareto frontier, we want to maximize accuracy and minimize latency
    # So we'll plot latency on x-axis (lower is better) and accuracy on y-axis (higher is better)
    
    # Plot all models
    for idx, row in df.iterrows():
        model_color = model_colors.get(row['model'], COLORS[0])
        ax.scatter(row['avg_latency'], row['accuracy'], s=300, alpha=0.7, 
                  color=model_color, label=row['model'], edgecolors='black', linewidth=1.5)
        # Adjust annotation position to avoid legend area and description
        # Place labels based on position
        if row['avg_latency'] < df['avg_latency'].median():
            xytext_offset = (10, 5)  # Right side
        else:
            xytext_offset = (-10, 5)  # Left side
        
        ax.annotate(row['model'], (row['avg_latency'], row['accuracy']),
                   xytext=xytext_offset, textcoords='offset points', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Find Pareto-optimal points (not dominated by any other point)
    # A point is Pareto-optimal if no other point has both lower latency AND higher accuracy
    pareto_points = []
    for idx1, row1 in df.iterrows():
        is_dominated = False
        for idx2, row2 in df.iterrows():
            if idx1 != idx2:
                # Check if row2 dominates row1 (lower latency AND higher accuracy)
                if row2['avg_latency'] < row1['avg_latency'] and row2['accuracy'] > row1['accuracy']:
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_points.append((row1['avg_latency'], row1['accuracy'], row1['model']))
    
    # Sort Pareto points by latency for drawing the frontier
    pareto_points.sort(key=lambda x: x[0])
    
    # Draw Pareto frontier line
    if len(pareto_points) > 1:
        pareto_latencies = [p[0] for p in pareto_points]
        pareto_accuracies = [p[1] for p in pareto_points]
        ax.plot(pareto_latencies, pareto_accuracies, 'r--', linewidth=2, alpha=0.7, 
               label='Pareto Frontier', zorder=0)
        
        # Highlight Pareto-optimal points
        for lat, acc, model in pareto_points:
            ax.scatter(lat, acc, s=500, color='red', marker='*', 
                      edgecolors='black', linewidth=2, zorder=10, alpha=0.9)
    
    # Add shaded area showing dominated region (approximate)
    if len(pareto_points) > 0:
        # Find the "best" point (highest accuracy, lowest latency among Pareto points)
        best_point = max(pareto_points, key=lambda x: (x[1], -x[0]))
        ax.axvspan(0, best_point[0], alpha=0.1, color='green', label='Efficient Region')
        ax.axhspan(best_point[1], 1, alpha=0.1, color='green')
    
    ax.set_xlabel('Average Latency (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Pareto Frontier: Efficiency Analysis\n(Points on frontier are not dominated)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(alpha=0.3)
    # Move legend to lower right to avoid overlapping with description text
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9, 
             ncol=1, columnspacing=1)
    
    # Add text explaining Pareto frontier - move to lower left to avoid legend
    ax.text(0.02, 0.15, 
           'Pareto Frontier: Models where no other model\nhas both lower latency AND higher accuracy',
           transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created pareto_frontier.png")

def plot_per_email_statistics(per_email_df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create visualization of per-email accuracy statistics."""
    if per_email_df.empty:
        print("⚠ Skipping per-email statistics (no data)")
        return
    
    if model_colors is None:
        model_colors = get_model_color_map(per_email_df['model'].unique())
    
    # Calculate model ordering by success rate (for consistent sorting across all panels)
    # Calculate success rate per model - select status column first to avoid FutureWarning
    model_success_rates = per_email_df.groupby('model')['status'].apply(
        lambda x: len(x[x.isin(['TP', 'TN'])]) / len(x) if len(x) > 0 else 0
    )
    # The result is a Series with model as index, so we can sort directly
    model_success_rates = model_success_rates.sort_values(ascending=False)
    models_sorted = model_success_rates.index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Per-Email Accuracy Statistics\n(Models ordered by success rate)', fontsize=16, fontweight='bold')
    
    # 1. Score distribution: Correct vs Incorrect predictions (shows model calibration)
    ax1 = axes[0, 0]
    score_bins = get_score_bins()
    for model in models_sorted:
        model_data = per_email_df[per_email_df['model'] == model]
        correct_scores = model_data[model_data['correct'] == True]['score']
        incorrect_scores = model_data[model_data['correct'] == False]['score']
        model_color = model_colors.get(model, COLORS[0])
        
        ax1.hist(correct_scores, alpha=0.5, label=f'{model} (Correct)', bins=score_bins, 
                edgecolor='black', linewidth=0.5, color=model_color, align='mid')
        # Use darker version for incorrect
        import matplotlib.colors as mcolors
        darker_color = tuple(c * 0.6 for c in model_color[:3]) if len(model_color) >= 3 else model_color
        ax1.hist(incorrect_scores, alpha=0.5, label=f'{model} (Incorrect)', bins=score_bins,
                edgecolor='black', linewidth=0.5, color=darker_color, hatch='///', align='mid')
    
    ax1.set_xlabel('Score (0-100)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distribution: Correct vs Incorrect Predictions\n(Shows model calibration)')
    ax1.set_xlim(*get_score_axis_limits())  # Set explicit limits to show 0-100 range
    ax1.legend(fontsize=8, loc='upper right', ncol=2)
    ax1.grid(alpha=0.3, axis='y')
    
    # 2. Score distribution by model
    ax2 = axes[0, 1]
    score_bins = get_score_bins()  # Use same aligned bins for consistency
    for model in models_sorted:
        model_scores = per_email_df[per_email_df['model'] == model]['score']
        model_color = model_colors.get(model, COLORS[0])
        ax2.hist(model_scores, alpha=0.5, label=model, bins=score_bins, 
                edgecolor='black', linewidth=0.5, align='mid', color=model_color)
    ax2.set_xlabel('Score (0-100)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution by Model')
    ax2.set_xlim(*get_score_axis_limits())  # Set explicit limits to show 0-100 range
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3, axis='y')
    
    # 3. Status breakdown (TP, TN, FP, FN, TIMEOUT, ERROR) by model
    ax3 = axes[1, 0]
    status_counts = per_email_df.groupby(['model', 'status']).size().unstack(fill_value=0)
    # Reorder status_counts by success rate (already calculated above)
    status_counts = status_counts.reindex(model_success_rates.index)
    
    # Define color mapping for all possible statuses
    status_colors = {
        'TP': 'green',
        'TN': 'blue', 
        'FP': 'orange',
        'FN': 'red',
        'TIMEOUT': 'gray',
        'JSON_ERROR': 'purple',
        'ERROR': 'darkred'
    }
    
    # Get all statuses and assign colors
    all_statuses = status_counts.columns.tolist()
    colors = [status_colors.get(status, 'lightgray') for status in all_statuses]
    
    status_counts.plot(kind='barh', stacked=True, ax=ax3, color=colors)
    ax3.set_xlabel('Count')
    ax3.set_title('Prediction Status Breakdown by Model\n(Includes timeouts/errors)')
    ax3.legend(title='Status', fontsize=8)
    ax3.grid(axis='x', alpha=0.3)
    
    # Add note about timeouts/errors if any model has them
    error_statuses = ['TIMEOUT', 'JSON_ERROR', 'ERROR']
    has_errors = any(status in all_statuses for status in error_statuses)
    if has_errors:
        error_models = per_email_df[per_email_df['status'].isin(error_statuses)]['model'].unique()
        if len(error_models) > 0:
            ax3.text(0.98, 0.02, f'WARNING: {len(error_models)} model(s) have timeouts/errors', 
                    transform=ax3.transAxes, fontsize=8, ha='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 4. Duration distribution by model
    ax4 = axes[1, 1]
    max_duration = per_email_df['duration'].max()
    duration_bins = get_duration_bins(max_duration)
    for model in models_sorted:
        model_durations = per_email_df[per_email_df['model'] == model]['duration']
        model_color = model_colors.get(model, COLORS[0])
        ax4.hist(model_durations, alpha=0.5, label=model, bins=duration_bins, 
                edgecolor='black', linewidth=0.5, align='mid', color=model_color)
    ax4.set_xlabel('Inference Duration (seconds)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Inference Duration Distribution by Model')
    # Set explicit limits to show actual range (excluding the extra bin at the end)
    ax4.set_xlim(duration_bins[0], duration_bins[-1] - DURATION_BIN_WIDTH)
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_email_statistics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created per_email_statistics.png")

def plot_reason_consistency_by_type(repeat_df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create visualization of reason consistency broken down by email type (benign vs malicious)."""
    if repeat_df.empty:
        print("⚠ Skipping reason consistency by type (no data)")
        return
    
    if model_colors is None:
        model_colors = get_model_color_map(list(repeat_df['model'].unique()))
    
    # Map ground_truth: 0 = benign, 1 = malicious
    repeat_df_copy = repeat_df.copy()
    repeat_df_copy['email_type'] = repeat_df_copy['ground_truth'].map({0: 'Benign', 1: 'Malicious'})
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Reason Consistency by Email Type', fontsize=14, fontweight='bold')
    
    # 1. Average reason consistency by model and email type
    ax1 = axes[0]
    consistency_by_type = repeat_df_copy.groupby(['model', 'email_type'])['reason_consistency'].mean().unstack(fill_value=0)
    # Sort by overall consistency (average of benign and malicious) - higher is better
    overall_consistency = consistency_by_type.mean(axis=1).sort_values(ascending=False)
    consistency_by_type = consistency_by_type.reindex(overall_consistency.index)
    # Use grouped bar chart with model-specific colors
    x = np.arange(len(consistency_by_type.index))
    width = 0.35
    models = consistency_by_type.index
    for i, model in enumerate(models):
        model_color = model_colors.get(model, COLORS[0])
        benign_val = consistency_by_type.loc[model, 'Benign'] if 'Benign' in consistency_by_type.columns else 0
        malicious_val = consistency_by_type.loc[model, 'Malicious'] if 'Malicious' in consistency_by_type.columns else 0
        ax1.bar(x[i] - width/2, benign_val, width, color=model_color, alpha=0.6, label='Benign' if i == 0 else '')
        ax1.bar(x[i] + width/2, malicious_val, width, color=model_color, alpha=0.9, label='Malicious' if i == 0 else '')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Reason Consistency')
    ax1.set_title('Reason Consistency: Benign vs Malicious Emails')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.set_ylim(0, 1.1)
    ax1.legend(title='Email Type', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Score variance by model and email type
    ax2 = axes[1]
    variance_by_type = repeat_df_copy.groupby(['model', 'email_type'])['score_variance'].mean().unstack(fill_value=0)
    # Sort by overall variance (lower is better) - use same order as consistency chart
    variance_by_type = variance_by_type.reindex(overall_consistency.index)
    # Use grouped bar chart with model-specific colors
    x = np.arange(len(variance_by_type.index))
    models = variance_by_type.index
    for i, model in enumerate(models):
        model_color = model_colors.get(model, COLORS[0])
        benign_val = variance_by_type.loc[model, 'Benign'] if 'Benign' in variance_by_type.columns else 0
        malicious_val = variance_by_type.loc[model, 'Malicious'] if 'Malicious' in variance_by_type.columns else 0
        ax2.bar(x[i] - width/2, benign_val, width, color=model_color, alpha=0.6, label='Benign' if i == 0 else '')
        ax2.bar(x[i] + width/2, malicious_val, width, color=model_color, alpha=0.9, label='Malicious' if i == 0 else '')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Average Score Variance')
    ax2.set_title('Score Variance: Benign vs Malicious Emails')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend(title='Email Type', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reason_consistency_by_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created reason_consistency_by_type.png")


def plot_temperature_impact(df: pd.DataFrame, output_dir: str, model_colors: Dict[str, tuple] = None):
    """Create visualization showing impact of temperature settings on accuracy metrics (not repeatability)."""
    if df.empty:
        print("⚠ Skipping temperature impact (no data)")
        return
    
    if model_colors is None:
        model_colors = get_model_color_map(list(df['model'].unique()))
    
    # Check if we have temperature data
    if 'temperature' not in df.columns or df['temperature'].isna().all():
        print("⚠ Skipping temperature impact (no temperature data)")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Impact of Temperature on Accuracy Metrics', fontsize=14, fontweight='bold')
    
    # 1. Temperature vs accuracy
    ax1 = axes[0]
    temp_accuracy = df.groupby(['model', 'temperature']).agg({
        'accuracy': 'mean'
    }).reset_index()
    
    for model in temp_accuracy['model'].unique():
        model_data = temp_accuracy[temp_accuracy['model'] == model]
        model_color = model_colors.get(model, COLORS[0])
        ax1.plot(model_data['temperature'], model_data['accuracy'], 
                marker='o', label=model, linewidth=2, markersize=8, color=model_color)
    
    ax1.set_xlabel('Temperature')
    ax1.set_ylabel('Average Accuracy')
    ax1.set_title('Temperature vs Accuracy')
    ax1.legend(fontsize=8, loc='best')
    ax1.grid(alpha=0.3)
    
    # 2. Temperature vs F1 score
    ax2 = axes[1]
    temp_f1 = df.groupby(['model', 'temperature']).agg({
        'f1': 'mean'
    }).reset_index()
    
    for model in temp_f1['model'].unique():
        model_data = temp_f1[temp_f1['model'] == model]
        model_color = model_colors.get(model, COLORS[0])
        ax2.plot(model_data['temperature'], model_data['f1'],
                marker='o', label=model, linewidth=2, markersize=8, color=model_color)
    
    ax2.set_xlabel('Temperature')
    ax2.set_ylabel('Average F1 Score')
    ax2.set_title('Temperature vs F1 Score')
    ax2.set_ylim(0, 1.1)
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_impact.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created temperature_impact.png")

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

def generate_summary_report(df: pd.DataFrame, repeat_df: pd.DataFrame, output_dir: str, per_email_df: pd.DataFrame = None):
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
            
            # Reason consistency by email type
            repeat_df_copy = repeat_df.copy()
            repeat_df_copy['email_type'] = repeat_df_copy['ground_truth'].map({0: 'Benign', 1: 'Malicious'})
            consistency_by_type = repeat_df_copy.groupby(['model', 'email_type'])['reason_consistency'].mean().unstack(fill_value=0)
            if not consistency_by_type.empty:
                f.write("### Reason Consistency by Email Type\n\n")
                f.write("This analysis shows whether models provide consistent reasoning for benign vs malicious emails:\n\n")
                f.write(dataframe_to_markdown(consistency_by_type))
                f.write("\n\n")
            
            # Model size correlation
            if 'model_size' in df.columns and df['model_size'].sum() > 0:
                model_sizes = df[['model', 'model_size']].set_index('model')
                repeat_with_size = repeat_df.merge(model_sizes, left_on='model', right_index=True, how='left')
                if not repeat_with_size.empty and repeat_with_size['model_size'].notna().any():
                    variance_by_size = repeat_with_size.groupby('model').agg({
                        'model_size': 'first',
                        'score_variance': 'mean'
                    }).reset_index()
                    consistency_by_size = repeat_with_size.groupby('model').agg({
                        'model_size': 'first',
                        'reason_consistency': 'mean'
                    }).reset_index()
                    
                    if len(variance_by_size) > 1:
                        # Check for sufficient variance before calculating correlation
                        var_corr = None
                        cons_corr = None
                        
                        # Suppress numpy warnings for correlation calculation
                        with np.errstate(divide='ignore', invalid='ignore'):
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning)
                                
                                # Only calculate if there's variance in both variables
                                if (variance_by_size['model_size'].std() > 0 and 
                                    variance_by_size['score_variance'].std() > 0):
                                    try:
                                        var_corr = variance_by_size['model_size'].corr(variance_by_size['score_variance'])
                                    except (ValueError, ZeroDivisionError):
                                        pass
                                
                                if (consistency_by_size['model_size'].std() > 0 and 
                                    consistency_by_size['reason_consistency'].std() > 0):
                                    try:
                                        cons_corr = consistency_by_size['model_size'].corr(consistency_by_size['reason_consistency'])
                                    except (ValueError, ZeroDivisionError):
                                        pass
                        
                        f.write("### Model Size vs Repeatability Correlation\n\n")
                        f.write("Analysis of whether larger models show different repeatability patterns:\n\n")
                        if var_corr is not None and pd.notna(var_corr) and not np.isnan(var_corr):
                            f.write(f"- **Model Size vs Score Variance Correlation**: {var_corr:.3f}\n")
                        if cons_corr is not None and pd.notna(cons_corr) and not np.isnan(cons_corr):
                            f.write(f"- **Model Size vs Reason Consistency Correlation**: {cons_corr:.3f}\n")
                        if (var_corr is None or (pd.isna(var_corr) or np.isnan(var_corr))) and \
                           (cons_corr is None or (pd.isna(cons_corr) or np.isnan(cons_corr))):
                            f.write("- Insufficient variance in model size or repeatability metrics to calculate meaningful correlations\n")
                        f.write("\n")
            
            # Temperature impact
            if 'temperature' in repeat_df.columns and repeat_df['temperature'].notna().any():
                temp_summary = repeat_df.groupby(['model', 'temperature']).agg({
                    'score_variance': 'mean',
                    'reason_consistency': 'mean'
                }).reset_index()
                if len(temp_summary['temperature'].unique()) > 1:
                    f.write("### Temperature Impact on Consistency\n\n")
                    f.write("Analysis of how temperature settings affect model consistency:\n\n")
                    f.write(dataframe_to_markdown(temp_summary))
                    f.write("\n\n")
        
        f.write("## Recommendations\n\n")
        
        # Generate recommendations based on metrics
        if best_accuracy['fpr'] < 0.15:
            f.write(f"- **For Production Use**: {best_accuracy['model']} offers the best balance of accuracy with low false positive rate\n")
        
        if best_recall['fnr'] < 0.15:
            f.write(f"- **For High-Security Environments**: {best_recall['model']} minimizes false negatives (missed threats)\n")
        
        if fastest['avg_latency'] < 2.0:
            f.write(f"- **For Real-Time Applications**: {fastest['model']} provides the fastest inference suitable for browser-based detection\n")
        
        # Per-email statistics summary
        f.write("## Per-Email Statistics Analysis\n\n")
        f.write("This section analyzes individual email-level performance patterns:\n\n")
        
        # Check for timeout/error issues
        if per_email_df is not None and not per_email_df.empty:
            error_statuses = ['TIMEOUT', 'JSON_ERROR', 'ERROR']
            error_models = per_email_df[per_email_df['status'].isin(error_statuses)]['model'].unique()
            if len(error_models) > 0:
                f.write("### WARNING: Timeout/Error Warnings\n\n")
                f.write("The following models experienced timeouts or errors during evaluation:\n\n")
                for model in error_models:
                    model_data = per_email_df[per_email_df['model'] == model]
                    total = len(model_data)
                    timeouts = len(model_data[model_data['status'] == 'TIMEOUT'])
                    json_errors = len(model_data[model_data['status'] == 'JSON_ERROR'])
                    other_errors = len(model_data[model_data['status'] == 'ERROR'])
                    total_errors = timeouts + json_errors + other_errors
                    successful = total - total_errors
                    
                    error_details = []
                    if timeouts > 0:
                        error_details.append(f"{timeouts} timeouts")
                    if json_errors > 0:
                        error_details.append(f"{json_errors} JSON parsing errors")
                    if other_errors > 0:
                        error_details.append(f"{other_errors} other errors")
                    
                    f.write(f"- **{model}**: {', '.join(error_details)} out of {total} total ({successful} successful, {(total_errors/total*100):.1f}% failed)\n")
                    if json_errors > 0:
                        f.write(f"  - JSON parsing errors indicate the model output was not valid JSON format\n")
                    if timeouts > 0:
                        f.write(f"  - Timeouts suggest the model is too slow for production use\n")
                    f.write(f"  - **Important**: Failed predictions are not included in TP/TN/FP/FN counts, which can make accuracy metrics misleading\n")
                    f.write(f"  - Consider increasing timeout limits, fixing JSON output format, or optimizing model performance for future benchmarks\n\n")
                f.write("\n")
        
        f.write("### Key Findings from Per-Email Data:\n\n")
        f.write("- **Score Distribution**: Shows model confidence patterns and calibration\n")
        f.write("  - Well-calibrated models show higher scores for correct predictions\n")
        f.write("  - Overconfident models show high scores even when incorrect\n")
        f.write("- **Status Breakdown**: Detailed TP, TN, FP, FN, TIMEOUT, JSON_ERROR, ERROR counts per model\n")
        f.write("  - Reveals the composition of errors (false positives vs false negatives)\n")
        f.write("  - TIMEOUT: Model exceeded time limit (typically 60 seconds)\n")
        f.write("  - JSON_ERROR: Model output was not valid JSON format\n")
        f.write("  - ERROR: Other types of errors (network, API, etc.)\n")
        f.write("  - These error cases indicate incomplete evaluations that should be addressed\n")
        f.write("- **Duration Distribution**: Shows latency variability across emails\n")
        f.write("  - Identifies if inference time is consistent or has outliers\n")
        f.write("\n")
        f.write("See `per_email_statistics.png` for detailed visualizations.\n\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("The following visualizations are available:\n")
        f.write("- `accuracy_comparison.png`: Overall accuracy, precision/recall, F1, and error rates\n")
        f.write("- `confusion_matrices.png`: Confusion matrices for all models\n")
        f.write("- `latency_comparison.png`: Inference speed comparison\n")
        f.write("- `repeatability_analysis.png`: Consistency and repeatability metrics\n")
        f.write("- `reason_consistency_by_type.png`: Reason consistency broken down by email type (benign vs malicious)\n")
        f.write("- `temperature_impact.png`: Impact of temperature settings on accuracy metrics\n")
        f.write("- `per_email_statistics.png`: Per-email accuracy statistics and distributions\n")
        f.write("- `radar_chart.png`: Multi-metric radar comparison\n")
        f.write("- `latency_accuracy_tradeoff.png`: Latency vs accuracy tradeoff scatter plot\n")
        f.write("- `quadrant_analysis.png`: Four-quadrant analysis categorizing models\n")
        f.write("- `metrics_heatmap.png`: Comprehensive metrics heatmap\n")
        f.write("- `pareto_frontier.png`: Pareto efficiency frontier analysis\n")
    
    print(f"✓ Created ANALYSIS_REPORT.md")

def main():
    """Main analysis pipeline."""
    # Setup paths
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark-results"
    output_dir = script_dir
    
    print("🔍 Loading benchmark results...")
    # Load accuracy results (exclude repeatability files)
    accuracy_results = load_benchmark_results(str(results_dir), include_repeatability=False)
    # Load repeatability results (only files with '_repeatability_' in name)
    repeatability_results = load_benchmark_results(str(results_dir), include_repeatability=True)
    
    if not accuracy_results:
        print("❌ No accuracy benchmark results found!")
        return
    
    print(f"✓ Loaded {len(accuracy_results)} accuracy benchmark result files")
    if repeatability_results:
        print(f"✓ Loaded {len(repeatability_results)} repeatability benchmark result files")
    print()
    
    # Extract metrics
    print("📊 Extracting metrics...")
    df = extract_summary_metrics(accuracy_results)
    repeat_df = extract_repeatability_metrics(repeatability_results)
    per_email_df = extract_per_email_metrics(accuracy_results)
    
    # Add error rate metrics to summary dataframe
    if not per_email_df.empty:
        error_rates = []
        for model in df['model'].unique():
            model_data = per_email_df[per_email_df['model'] == model]
            total = len(model_data)
            if total > 0:
                timeouts = len(model_data[model_data['status'] == 'TIMEOUT'])
                json_errors = len(model_data[model_data['status'] == 'JSON_ERROR'])
                other_errors = len(model_data[model_data['status'] == 'ERROR'])
                total_errors = timeouts + json_errors + other_errors
                
                error_rates.append({
                    'model': model,
                    'timeout_rate': timeouts / total,
                    'json_error_rate': json_errors / total,
                    'error_rate': other_errors / total,
                    'total_error_rate': total_errors / total,
                    'success_rate': (total - total_errors) / total
                })
            else:
                error_rates.append({
                    'model': model,
                    'timeout_rate': 0,
                    'json_error_rate': 0,
                    'error_rate': 0,
                    'total_error_rate': 0,
                    'success_rate': 1.0
                })
        
        error_rates_df = pd.DataFrame(error_rates)
        df = df.merge(error_rates_df, on='model', how='left')
        # Fill NaN with 0 for models without per-email data
        error_cols = ['timeout_rate', 'json_error_rate', 'error_rate', 'total_error_rate', 'success_rate']
        df[error_cols] = df[error_cols].fillna(0)
    
    print(f"✓ Extracted metrics for {len(df)} models")
    print(f"  Models: {sorted(df['model'].unique())}\n")
    
    # Create visualizations directory
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Create consistent color mapping for all models (both accuracy and repeatability)
    # Since file names are standardized, we can match models by base name
    # For repeatability models, match them to accuracy models if they have the same base name
    def get_base_model_name(name: str) -> str:
        """Get base model name for matching.
        Removes -reasoning suffix and version numbers (like -2057, -2507) for matching."""
        import re
        # Remove -reasoning suffix if present
        if name.endswith('-reasoning'):
            name = name[:-10]  # Remove '-reasoning'
        # Remove trailing version numbers (4+ digits after dash, like -2057, -2507)
        # This matches models like qwen_qwen3-4b-2057 and qwen_qwen3-4b-2507
        name = re.sub(r'-(\d{4,})$', '', name)  # Remove trailing 4+ digit version numbers
        return name
    
    # Collect ALL unique models (both accuracy and repeatability) for color mapping
    all_models = list(df['model'].unique())
    if not repeat_df.empty:
        for model in repeat_df['model'].unique():
            if model not in all_models:
                all_models.append(model)
    
    # Create color map using ALL original model names
    all_models_sorted = sorted(all_models)
    model_colors = get_model_color_map(all_models_sorted)
    
    # For models that match between accuracy and repeatability (same base name), use same color
    if not repeat_df.empty:
        accuracy_models = set(df['model'].unique())
        repeatability_models = set(repeat_df['model'].unique())
        
        # Create mapping: base name -> list of original names that match
        base_to_models = {}
        for model in accuracy_models:
            base = get_base_model_name(model)
            if base not in base_to_models:
                base_to_models[base] = []
            base_to_models[base].append(('accuracy', model))
        
        for model in repeatability_models:
            base = get_base_model_name(model)
            if base not in base_to_models:
                base_to_models[base] = []
            base_to_models[base].append(('repeatability', model))
        
        # For each base name that has both accuracy and repeatability models, use same color
        for base, model_list in base_to_models.items():
            acc_models = [m for t, m in model_list if t == 'accuracy']
            rep_models = [m for t, m in model_list if t == 'repeatability']
            
            if acc_models and rep_models:
                # Use the accuracy model's color for the repeatability model
                acc_color = model_colors.get(acc_models[0], COLORS[0])
                for rep_model in rep_models:
                    model_colors[rep_model] = acc_color
    
    print(f"✓ Created color mapping for {len(model_colors)} models")
    print(f"  Accuracy models ({len(df['model'].unique())}): {sorted(df['model'].unique())}")
    if not repeat_df.empty:
        print(f"  Repeatability models ({len(repeat_df['model'].unique())}): {sorted(repeat_df['model'].unique())}")
        # Check for matching models (by base name)
        accuracy_base = {get_base_model_name(m): m for m in df['model'].unique()}
        repeatability_base = {get_base_model_name(m): m for m in repeat_df['model'].unique()}
        matching_base = set(accuracy_base.keys()) & set(repeatability_base.keys())
        if matching_base:
            print(f"  Matching models (same color):")
            for base_name in sorted(matching_base):
                acc_models = [m for m in df['model'].unique() if get_base_model_name(m) == base_name]
                rep_models = [m for m in repeat_df['model'].unique() if get_base_model_name(m) == base_name]
                for acc_model in acc_models:
                    for rep_model in rep_models:
                        print(f"    - '{acc_model}' (accuracy) <-> '{rep_model}' (repeatability)")
    print()
    
    # Generate visualizations
    print("📈 Creating visualizations...")
    plot_accuracy_comparison(df, viz_dir, model_colors)
    plot_confusion_matrix_comparison(df, viz_dir)
    plot_latency_comparison(df, viz_dir, model_colors)
    if not repeat_df.empty:
        plot_repeatability_analysis(repeat_df, viz_dir, model_colors)
        plot_reason_consistency_by_type(repeat_df, viz_dir, model_colors)
    # Temperature impact should use accuracy results, not repeatability
    plot_temperature_impact(df, viz_dir, model_colors)
    plot_radar_chart(df, viz_dir, model_colors)
    plot_latency_accuracy_tradeoff(df, viz_dir, model_colors)
    plot_quadrant_analysis(df, viz_dir, model_colors)
    plot_metrics_heatmap(df, viz_dir)
    plot_pareto_frontier(df, viz_dir, model_colors)
    if not per_email_df.empty:
        plot_per_email_statistics(per_email_df, viz_dir, model_colors)
    
    print("\n📝 Generating summary report...")
    generate_summary_report(df, repeat_df, str(output_dir), per_email_df)
    
    # Save raw data as CSV
    df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    if not repeat_df.empty:
        repeat_df.to_csv(os.path.join(output_dir, 'repeatability_metrics.csv'), index=False)
    if not per_email_df.empty:
        per_email_df.to_csv(os.path.join(output_dir, 'per_email_metrics.csv'), index=False)
    
    print("\n✅ Analysis complete! Check the benchmark-analysis folder for results.")
    print(f"\n📁 Output files:")
    print(f"   - ANALYSIS_REPORT.md")
    print(f"   - summary_metrics.csv")
    print(f"   - repeatability_metrics.csv")
    if not per_email_df.empty:
        print(f"   - per_email_metrics.csv")
    print(f"   - visualizations/*.png (all visualization images)")

if __name__ == "__main__":
    main()

