"""
Test evaluation module with actual CSV dataset.

This script loads the actual phishing email dataset and tests model outputs against it.
"""

import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
script_dir = Path(__file__).parent.absolute()
parent_dir = script_dir.parent
sys.path.insert(0, str(parent_dir))

from evaluation import evaluate_model_response, evaluate_batch, evaluate_from_dataset, print_evaluation_report

# Dataset path
DATASET_PATH = parent_dir / "dataset" / "phishing_email.csv"

def load_sample_data(n_samples=10):
    """Load a balanced sample of data from the dataset."""
    print(f"Loading dataset from: {DATASET_PATH}")
    
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    # Load full dataset to get balanced sample
    df_full = pd.read_csv(DATASET_PATH)
    
    # Get balanced sample (half phishing, half legitimate)
    phishing_samples = df_full[df_full['label'] == 1].head(n_samples // 2)
    legitimate_samples = df_full[df_full['label'] == 0].head(n_samples // 2)
    
    # Combine and shuffle
    df = pd.concat([phishing_samples, legitimate_samples]).sample(frac=1).reset_index(drop=True)
    
    print(f"Loaded {len(df)} samples (balanced)")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    return df

def create_sample_model_outputs(df):
    """
    Create sample model outputs based on the dataset.
    In a real scenario, these would come from your SLM model.
    
    Label mapping: 0 = legitimate/not_malicious, 1 = phishing/malicious
    """
    model_outputs = []
    
    for idx, row in df.iterrows():
        label = row['label']
        text = row['text_combined']
        
        # Simulate model prediction based on label (in real use, model would analyze text)
        # For testing, we'll create some correct and some incorrect predictions
        if idx % 3 == 0:
            # Every 3rd prediction is incorrect for testing
            predicted_label = "malicious" if label == 0 else "not_malicious"
        else:
            # Correct predictions
            predicted_label = "malicious" if label == 1 else "not_malicious"
        
        # Generate sample reasons based on the prediction
        if predicted_label == "malicious":
            reasons = [
                "contains suspicious patterns",
                "potential phishing indicators detected"
            ]
        else:
            reasons = [
                "appears to be legitimate communication",
                "no obvious phishing indicators"
            ]
        
        model_output = {
            "label": predicted_label,
            "reasons": reasons
        }
        
        model_outputs.append(model_output)
    
    return model_outputs

def convert_labels_to_strings(df):
    """Convert numeric labels (0, 1) to string labels."""
    # Map: 0 = legitimate/not_malicious, 1 = phishing/malicious
    label_mapping = {0: "not_malicious", 1: "malicious"}
    return df['label'].map(label_mapping).tolist()

def main():
    print("=" * 60)
    print("TESTING EVALUATION WITH ACTUAL DATASET")
    print("=" * 60)
    print()
    
    # Load sample data
    try:
        df = load_sample_data(n_samples=20)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("\n" + "-" * 60)
    print("Sample Data Preview:")
    print("-" * 60)
    for i in range(min(3, len(df))):
        label = df.iloc[i]['label']
        text = df.iloc[i]['text_combined']
        print(f"\nSample {i+1}:")
        print(f"  Label: {label} ({'phishing' if label == 1 else 'legitimate'})")
        print(f"  Text preview: {text[:100]}...")
    
    # Create sample model outputs
    print("\n" + "-" * 60)
    print("Creating Sample Model Outputs...")
    print("-" * 60)
    model_outputs = create_sample_model_outputs(df)
    
    # Convert ground truth labels
    ground_truth_labels = convert_labels_to_strings(df)
    
    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    results = evaluate_batch(
        model_outputs,
        ground_truth_labels,
        label_mapping=None  # Labels are already normalized
    )
    
    print_evaluation_report(results, detailed=True)
    
    # Test individual evaluation
    print("\n" + "=" * 60)
    print("INDIVIDUAL SAMPLE EVALUATION")
    print("=" * 60)
    
    for i in range(min(5, len(model_outputs))):
        print(f"\nSample {i+1}:")
        print(f"  Email text: {df.iloc[i]['text_combined'][:80]}...")
        is_correct, details = evaluate_model_response(
            model_outputs[i],
            ground_truth_labels[i]
        )
        print(f"  Predicted: {details['predicted']}")
        print(f"  Ground Truth: {details['ground_truth']}")
        print(f"  Correct: {is_correct} {'[PASS]' if is_correct else '[FAIL]'}")
        print(f"  Reasons: {', '.join(details['reasons'])}")
    
    # Test with evaluate_from_dataset function
    print("\n" + "=" * 60)
    print("TESTING evaluate_from_dataset FUNCTION")
    print("=" * 60)
    
    try:
        # Use a smaller sample for this test
        small_df = df.head(10)
        small_model_outputs = model_outputs[:10]
        
        # Save a temporary CSV for testing
        temp_csv = parent_dir / "dataset" / "temp_test_sample.csv"
        small_df.to_csv(temp_csv, index=False)
        
        # Convert numeric labels to strings first (use copy to avoid warning)
        small_df_copy = small_df.copy()
        small_df_copy['label_str'] = small_df_copy['label'].map({0: "not_malicious", 1: "malicious"})
        small_df_copy[['text_combined', 'label_str']].to_csv(temp_csv, index=False)
        
        results_from_file = evaluate_from_dataset(
            small_model_outputs,
            temp_csv,
            label_column="label_str",
            label_mapping=None
        )
        
        print_evaluation_report(results_from_file, detailed=False)
        
        # Clean up
        temp_csv.unlink()
        print("\nTemporary test file cleaned up.")
        
    except Exception as e:
        print(f"Error testing evaluate_from_dataset: {e}")

if __name__ == "__main__":
    main()

