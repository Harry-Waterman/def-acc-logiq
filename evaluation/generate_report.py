import argparse
import json
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib or seaborn not installed. Skipping image generation.")

def load_results(results_dir):
    results = {}
    dir_path = Path(results_dir)
    if not dir_path.exists():
        return results
        
    for file_path in dir_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Basic validation
                if "accuracy_results" in data:
                    results[file_path.name] = data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return results

def generate_markdown_report(results, output_file):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"# TinyRod Evaluation Report\n"
    md += f"**Generated:** {timestamp}\n\n"
    
    md += "## 1. Summary Metrics\n\n"
    md += "| Experiment | Accuracy | Precision | Recall | F1 Score | FPR | FNR | Latency (s) |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    
    for filename, data in results.items():
        res = data['accuracy_results']
        name = filename.replace('.json', '').replace('results_', '').replace('exp', 'Exp ')
        
        md += f"| {name} | {res['accuracy']:.2%} | {res['precision']:.2%} | {res['recall']:.2%} | {res['f1']:.2%} | {res['fpr']:.2%} | {res['fnr']:.2%} | {res['avg_latency']:.2f} |\n"
        
    md += "\n## 2. detailed Confusion Matrices\n\n"
    
    for filename, data in results.items():
        res = data['accuracy_results']
        name = filename.replace('.json', '')
        
        md += f"### {name}\n"
        md += f"- **Total Samples**: {res['total']}\n"
        md += f"- **Correct**: {res['correct']}\n\n"
        
        md += "| | Predicted Benign | Predicted Malicious |\n"
        md += "|---|---|---|\n"
        md += f"| **Actual Benign** | TN: {res['tn']} | FP: {res['fp']} |\n"
        md += f"| **Actual Malicious** | FN: {res['fn']} | TP: {res['tp']} |\n\n"
        
        if PLOTTING_AVAILABLE:
            img_name = f"cm_{name}.png"
            md += f"![Confusion Matrix]({img_name})\n\n"

    # Latency Section
    md += "\n## 3. Latency Analysis\n\n"
    if PLOTTING_AVAILABLE:
        md += "![Latency Distribution](latency_dist.png)\n\n"
        
    with open(output_file, 'w') as f:
        f.write(md)
    
    print(f"Report written to {output_file}")

def generate_plots(results, output_dir):
    if not PLOTTING_AVAILABLE:
        return
    
    sns.set_style("whitegrid")
    output_path = Path(output_dir)
    
    # 1. Confusion Matrices
    for filename, data in results.items():
        res = data['accuracy_results']
        name = filename.replace('.json', '')
        
        # Data: [[TN, FP], [FN, TP]]
        cm_data = [[res['tn'], res['fp']], [res['fn'], res['tp']]]
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Pred Benign', 'Pred Malicious'],
                    yticklabels=['Act Benign', 'Act Malicious'])
        plt.title(f"Confusion Matrix: {name}")
        plt.tight_layout()
        plt.savefig(output_path / f"cm_{name}.png")
        plt.close()
        
    # 2. Latency Distribution
    # Collect all latencies
    plt.figure(figsize=(10, 6))
    
    for filename, data in results.items():
        res = data['accuracy_results']
        if 'detailed_results' in res:
            latencies = [r['duration'] for r in res['detailed_results']]
            label = filename.replace('.json', '').replace('exp', 'Exp ')
            sns.kdeplot(latencies, label=label, fill=True, alpha=0.3)
            
    plt.title("Latency Distribution by Experiment")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "latency_dist.png")
    plt.close()
    
    print(f"Plots saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Visual Report for TinyRod Benchmarks")
    parser.add_argument("--results-dir", default="results", help="Directory containing JSON result files")
    parser.add_argument("--output", default="report.md", help="Output Markdown report file")
    
    args = parser.parse_args()
    
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No result files found.")
        return
        
    print(f"Found {len(results)} result files.")
    
    # Ensure output directory for images exists
    output_file_path = Path(args.output)
    output_dir = output_file_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # If output file is just a filename, save it in results dir by default if not specified
    if len(output_file_path.parts) == 1 and args.results_dir != ".":
         # Actually, keep user path. If they said "report.md", put it in CWD.
         pass

    generate_plots(results, output_dir)
    generate_markdown_report(results, args.output)

if __name__ == "__main__":
    main()

