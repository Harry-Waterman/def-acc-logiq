#!/usr/bin/env python3
"""Regenerate the metrics heatmap with fixed latency normalization."""
import json
import os
import glob
from pathlib import Path
import pandas as pd
from analyze_benchmarks import plot_metrics_heatmap, load_benchmark_results, extract_summary_metrics

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "benchmark-results"
    output_dir = script_dir / "visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading benchmark results...")
    results = load_benchmark_results(str(results_dir))
    
    if not results:
        print("No benchmark results found!")
        exit(1)
    
    print(f"Loaded {len(results)} benchmark result files")
    
    df = extract_summary_metrics(results)
    print(f"Extracted metrics for {len(df)} models")
    
    print("\nRegenerating metrics heatmap with fixed normalization...")
    plot_metrics_heatmap(df, str(output_dir))
    print("\n✅ Heatmap regenerated!")

