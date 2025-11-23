#!/bin/bash

# TinyRod Paper Benchmark Execution Script
# Runs the 3 key experiments defined in the upgrade plan.

# Ensure we are in the right directory
cd "$(dirname "$0")"

# Create results directory
mkdir -p results

echo "========================================================"
echo "Starting TinyRod Paper Benchmarks"
echo "========================================================"

# Experiment 1: Overall Performance (Mixed Dataset)
# Uses all available CSVs to create a balanced 50/50 split
echo ""
echo "[Experiment 1] Overall Performance (Mixed Dataset, n=400)"
python3 benchmark_all.py --sample-size 400 --output results/exp1_overall.json

# Experiment 2: False Positive Stress Test (Enron)
# Enron is typically all benign. This tests if we flag innocent emails.
echo ""
echo "[Experiment 2] False Positive Stress Test (Enron, n=400)"
# Note: If Enron is pure benign, this will result in ~200 benign samples (due to balancing logic capping at available data)
# unless we modify benchmark_all.py. Current behavior: returns min(200, len(benign)) + min(200, len(malicious)).
# For Enron (Benign only), we get 200 Benign.
python3 benchmark_all.py --dataset Enron.csv --sample-size 400 --output results/exp2_fp_stress.json

# Experiment 3: Obvious Phish Test (Nigerian Fraud)
# Nigerian Fraud is typically all malicious. This tests if we catch the obvious stuff.
echo ""
echo "[Experiment 3] Obvious Phish Test (Nigerian Fraud, n=400)"
# For Nigerian (Malicious only), we get 200 Malicious.
python3 benchmark_all.py --dataset Nigerian_Fraud.csv --sample-size 400 --output results/exp3_obvious_phish.json

echo ""
echo "========================================================"
echo "Benchmarks Complete."
echo "Results saved to evaluation/results/"
echo "Generating Visual Report..."
echo "========================================================"

# Run Visualizer
if [ -f "generate_report.py" ]; then
    python3 generate_report.py --results-dir results --output report.md
else
    echo "Warning: generate_report.py not found. Skipping visualization."
fi
