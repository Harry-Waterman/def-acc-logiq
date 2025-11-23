import argparse
import pandas as pd
import sys
from pathlib import Path
import time
import json
from collections import Counter
import statistics

import random

# Add current directory to path to ensure imports work if run from elsewhere
sys.path.append(str(Path(__file__).parent))

from dataset_loader import load_and_unify_datasets
from llm_client import LLMClient

def evaluate_accuracy(client: LLMClient, samples: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print(f"Running Accuracy Evaluation on {len(samples)} samples...")
    print("="*60)
    
    results = []
    
    # Confusion Matrix Counters
    tp = 0 # True Positive (Malicious detected as Malicious)
    tn = 0 # True Negative (Benign detected as Benign)
    fp = 0 # False Positive (Benign detected as Malicious)
    fn = 0 # False Negative (Malicious detected as Benign)
    
    latencies = []
    
    for idx, row in samples.iterrows():
        print(f"\nProcessing Sample {idx+1}/{len(samples)} (ID: {row.name})...")
        start_time = time.time()
        
        email_data = row.to_dict()
        ground_truth_label = int(row['label']) # 0 (Benign) or 1 (Malicious)
        
        response = client.classify_email(email_data)
        duration = time.time() - start_time
        latencies.append(duration)
        
        if "error" in response:
            print(f"  Error: {response['error']}")
            results.append({
                "index": idx,
                "ground_truth": ground_truth_label,
                "prediction": "ERROR",
                "score": None,
                "correct": False,
                "duration": duration,
                "error": response['error']
            })
            continue
            
        score = response.get('score', 0)
        # Normalize score to label: > 50 is Malicious (1), else Benign (0)
        predicted_label = 1 if score > 50 else 0
        
        is_correct = (predicted_label == ground_truth_label)
        
        # Update Confusion Matrix
        if ground_truth_label == 1 and predicted_label == 1:
            tp += 1
            status = "TP"
        elif ground_truth_label == 0 and predicted_label == 0:
            tn += 1
            status = "TN"
        elif ground_truth_label == 0 and predicted_label == 1:
            fp += 1
            status = "FP"
        elif ground_truth_label == 1 and predicted_label == 0:
            fn += 1
            status = "FN"
            
        print(f"  Result: {status} | Score: {score} | GT: {ground_truth_label} | Time: {duration:.2f}s")
        
        # Print sample context snippet
        print(f"  Source:  {email_data.get('source_file', 'Unknown')}")
        print(f"  Sender:  {email_data.get('sender', 'Unknown')}")
        print(f"  Subject: {email_data.get('subject', '')[:60]}")
        if email_data.get('attachment_names'):
            print(f"  Attachments: {email_data['attachment_names']}")
        print(f"  Body:    {str(email_data.get('body', ''))[:100]}...")
        
        if predicted_label == 1:
             print(f"  Reasons: {response.get('reasons', [])}")
             
        results.append({
            "index": idx,
            "ground_truth": ground_truth_label,
            "prediction": predicted_label,
            "score": score,
            "reasons": response.get('reasons', []),
            "correct": is_correct,
            "status": status,
            "duration": duration,
            "email_data": {
                "subject": email_data.get('subject', ''),
                "body_snippet": email_data.get('body', '')[:200],
                "sender": email_data.get('sender', ''),
                "source_file": email_data.get('source_file', 'Unknown')
            }
        })
        
    total = len(samples)
    correct_count = tp + tn
    accuracy = correct_count / total if total > 0 else 0
    
    # Derived Metrics
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall (TPR) = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # False Positive Rate (FPR) = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # False Negative Rate (FNR) = FN / (TP + FN)  (Same as 1 - Recall)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct_count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "avg_latency": avg_latency,
        "detailed_results": results
    }

def evaluate_repeatability(client: LLMClient, samples: pd.DataFrame, num_runs: int) -> dict:
    print("\n" + "="*60)
    print(f"Running Repeatability Evaluation on {len(samples)} samples ({num_runs} runs each)...")
    print("="*60)
    
    repeatability_stats = []
    
    for idx, row in samples.iterrows():
        print(f"\nTesting Stability for Sample {idx} (GT: {row['label']})...")
        print(f"  Source:  {row.get('source_file', 'Unknown')}")
        print(f"  Sender:  {row.get('sender', 'Unknown')}")
        print(f"  Subject: {str(row.get('subject', ''))[:60]}")
        if row.get('attachment_names'):
            print(f"  Attachments: {row['attachment_names']}")
        print(f"  Body:    {str(row.get('body', ''))[:100]}...")
        
        scores = []
        reasons_lists = []
        errors = 0
        
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...", end="\r")
            response = client.classify_email(row.to_dict())
            
            if "error" in response:
                errors += 1
                continue
                
            scores.append(response.get('score', 0))
            reasons_lists.append(tuple(sorted(response.get('reasons', [])))) # Tuple for hashing
            
        print(f"  Completed. Errors: {errors}")
        
        if not scores:
            continue
            
        # Calculate stats
        score_variance = statistics.variance(scores) if len(scores) > 1 else 0
        score_mean = statistics.mean(scores)
        
        # Reason stability (how many unique reason sets)
        unique_reason_sets = Counter(reasons_lists)
        most_common_reasons = unique_reason_sets.most_common(1)[0]
        reason_consistency = most_common_reasons[1] / len(scores) # % time we got the exact same reasons
        
        print(f"  Score Mean: {score_mean:.1f} | Variance: {score_variance:.1f}")
        print(f"  Reason Consistency: {reason_consistency:.1%}")
        
        repeatability_stats.append({
            "index": idx,
            "ground_truth": row['label'],
            "score_mean": score_mean,
            "score_variance": score_variance,
            "reason_consistency": reason_consistency,
            "scores": scores,
            "email_data": {
                "subject": row.get('subject', ''),
                "body_snippet": row.get('body', '')[:200],
                "sender": row.get('sender', ''),
                "source_file": row.get('source_file', 'Unknown')
            }
        })
        
    return repeatability_stats

def main():
    parser = argparse.ArgumentParser(description="Phishing Detection Benchmark Tool")
    parser.add_argument("--api-url", default="http://localhost:1234/v1", help="LLM API URL")
    parser.add_argument("--api-key", default="", help="API Key for external providers (e.g. OpenAI)")
    parser.add_argument("--model", default="local-model", help="Model name")
    parser.add_argument("--dataset", default=None, help="Specific dataset filename to load (e.g. Nazario.csv)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    parser.add_argument("--sample-size", type=int, default=0, help="Total samples for accuracy test")
    parser.add_argument("--repeat-runs", type=int, default=0, help="Number of runs for repeatability test")
    parser.add_argument("--only-repeatability", action="store_true", help="Skip accuracy test and run only repeatability")
    parser.add_argument("--retry-on-rate-limit", action="store_true", help="Retry on API 429 rate limits and timeouts")
    parser.add_argument("--data-dir", default=None, help="Directory containing CSV datasets")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        print(f"Using Random Seed: {args.seed}")
        random.seed(args.seed)
    else:
        # Generate a random seed for logging purposes
        args.seed = random.randint(0, 100000)
        print(f"Generated Random Seed: {args.seed}")
        random.seed(args.seed)
    
    # 1. Determine Data Directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Default to project root (2 levels up from here)
        data_dir = Path(__file__).parents[2]
        
    print(f"Loading datasets from: {data_dir}")
    
    try:
        df = load_and_unify_datasets(str(data_dir), filter_filename=args.dataset)
    except Exception as e:
        print(f"Fatal Error loading datasets: {e}")
        return

    # 2. Create Balanced Sample
    # We want 50% Benign (0) and 50% Malicious (1)
    benign_df = df[df['label'] == 0]
    malicious_df = df[df['label'] == 1]
    
    # FIX: If running only repeatability, ensure we have at least a small sample to pick from
    # We need at least 1 of each for the repeatability test below
    effective_sample_size = args.sample_size
    if args.only_repeatability and effective_sample_size < 2:
         effective_sample_size = 10 # Grab a small pool to pick from
    
    n_per_class = max(1, effective_sample_size // 2)
    
    # Sample with replacement if not enough data (though unlikely given dataset sizes)
    # Use random_state for reproducibility
    sample_benign = benign_df.sample(n=min(n_per_class, len(benign_df)), random_state=args.seed)
    # Use seed + 1 for malicious to ensure different randomness but deterministic
    sample_malicious = malicious_df.sample(n=min(n_per_class, len(malicious_df)), random_state=args.seed + 1)
    
    eval_df = pd.concat([sample_benign, sample_malicious]).sample(frac=1, random_state=args.seed + 2).reset_index(drop=True)
    
    print(f"\nPrepared Balanced Sample: {len(eval_df)} records")
    print(f"  Benign: {len(sample_benign)}")
    print(f"  Malicious: {len(sample_malicious)}")
    
    # 3. Initialize Client
    client = LLMClient(
        api_url=args.api_url, 
        api_key=args.api_key, 
        model=args.model, 
        temperature=args.temperature,
        retry_on_rate_limit=args.retry_on_rate_limit
    )
    
    # 4. Run Accuracy Test
    if not args.only_repeatability:
        acc_results = evaluate_accuracy(client, eval_df)
    else:
        print("\nSkipping Accuracy Test (--only-repeatability provided)")
        acc_results = {
            "accuracy": 0, "total": 0, "correct": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
            "precision": 0, "recall": 0, "f1": 0, "fpr": 0, "fnr": 0, "avg_latency": 0,
            "detailed_results": []
        }
    
    # 5. Run Repeatability Test
    # Pick 1 benign and 1 malicious from the sample
    rep_sample = pd.concat([
        sample_benign.head(1),
        sample_malicious.head(1)
    ])
    rep_results = evaluate_repeatability(client, rep_sample, args.repeat_runs)
    
    # 6. Final Report
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Accuracy:  {acc_results['accuracy']:.2%} ({acc_results['correct']}/{acc_results['total']})")
    print(f"Precision: {acc_results['precision']:.2%} (TP / (TP + FP))")
    print(f"Recall:    {acc_results['recall']:.2%} (TP / (TP + FN))")
    print(f"F1 Score:  {acc_results['f1']:.2%}")
    print(f"FPR:       {acc_results['fpr']:.2%} (False Positive Rate)")
    print(f"FNR:       {acc_results['fnr']:.2%} (False Negative Rate)")
    print(f"Avg Latency: {acc_results['avg_latency']:.2f}s")
    
    print("\nConfusion Matrix:")
    print("                 Predicted Benign (0) | Predicted Malicious (1)")
    print("Actual Benign (0)        {: <13} | {: <13}".format(f"TN: {acc_results['tn']}", f"FP: {acc_results['fp']}"))
    print("Actual Malicious (1)     {: <13} | {: <13}".format(f"FN: {acc_results['fn']}", f"TP: {acc_results['tp']}"))
    
    print(f"\nTotal Emails with Attachments in Sample: {eval_df[eval_df['attachment_names'].map(len) > 0].shape[0]}")
    
    print("\nRepeatability (Stability):")
    for r in rep_results:
        type_str = "Malicious" if r['ground_truth'] == 1 else "Benign"
        print(f"  {type_str} Sample:")
        print(f"    Score Variance: {r['score_variance']:.2f}")
        print(f"    Reason Consistency: {r['reason_consistency']:.1%}")
        
    # Save detailed results
    if args.output:
        output_file = args.output
    else:
        output_file = f"benchmark_results_seed{args.seed}_{int(time.time())}.json"
    
    # Ensure directory exists if output_file has a path
    output_path = Path(output_file)
    if output_path.parent != Path('.'):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create config dict but remove sensitive keys
    config_log = vars(args).copy()
    if "api_key" in config_log:
        config_log["api_key"] = "***"
        
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_log,
            "accuracy_results": acc_results,
            "repeatability_results": rep_results
        }, f, indent=2)
        
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()

