#!/usr/bin/env python3
"""Quick script to generate CSV files from benchmark results."""
import json
import os
import glob
from pathlib import Path
import pandas as pd

def load_benchmark_results(results_dir: str = "../benchmark-results"):
    """Load all benchmark JSON files from the results directory."""
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
    repeat_df = extract_repeatability_metrics(results)
    
    df.to_csv(os.path.join(output_dir, 'summary_metrics.csv'), index=False)
    print("✓ Created summary_metrics.csv")
    
    if not repeat_df.empty:
        repeat_df.to_csv(os.path.join(output_dir, 'repeatability_metrics.csv'), index=False)
        print("✓ Created repeatability_metrics.csv")
    else:
        print("⚠ No repeatability data available")

