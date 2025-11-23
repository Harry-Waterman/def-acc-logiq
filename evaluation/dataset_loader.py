import pandas as pd
import glob
import os
import re
from pathlib import Path
from typing import List, Optional

def load_and_unify_datasets(base_path: str, filter_filename: str = None) -> pd.DataFrame:
    """
    Loads all supported CSV datasets from the base path and unifies them into a single DataFrame.
    
    Args:
        base_path: Directory containing CSV files.
        filter_filename: Optional. If provided, only load the CSV matching this filename.
    
    Expected CSVs and their handling:
    1. Detailed Schema (Nigerian_Fraud, SpamAssasin, Nazario, CEAS_08?): 
       - sender, receiver, date, subject, body, urls, label
    2. Simple Schema (Enron, Ling):
       - subject, body, label
    3. Combined Schema (phishing_email):
       - text_combined, label
       
    Returns:
        pd.DataFrame: Unified dataframe with columns:
                      ['sender', 'receiver', 'date', 'subject', 'body', 'urls', 'label', 'source_file', 'attachment_names']
    """
    
    all_dfs = []
    
    # Regex for extracting potential attachments from body
    # Matches patterns like "attached file : name.ext", "filename=name.ext"
    # We'll reuse the compiled regex for efficiency
    attachment_regex = re.compile(r'(?:attached file\s*[:\-]?\s*|filename\s*[=:]\s*|name\s*[=:]\s*"?)([\w\-. ]+\.[a-zA-Z0-9]{2,4})(?:"|\s|$)', re.IGNORECASE)
    
    search_pattern = os.path.join(base_path, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    print(f"Found {len(csv_files)} CSV files in {base_path}")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Filter by filename if requested
        if filter_filename:
            filter_list = [f.strip() for f in filter_filename.split(',')]
            if filename not in filter_list:
                continue
            
        try:
            # specific handling for some files if they have encoding issues
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                except Exception as e:
                    print(f"Failed to read {filename}: {e}")
                    continue
            
            # Normalize columns to lowercase for easier matching
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Prepare standardized dataframe
            std_df = pd.DataFrame()
            std_df['source_file'] = [filename] * len(df)
            
            # Map Label (Always required)
            if 'label' in df.columns:
                std_df['label'] = df['label']
            else:
                print(f"Skipping {filename}: No 'label' column found.")
                continue
                
            # Map Body/Text and infer Subject/Sender if missing
            if 'body' in df.columns:
                std_df['body'] = df['body']
            elif 'text_combined' in df.columns:
                # For phishing_email.csv, text_combined contains everything
                std_df['body'] = df['text_combined']
                std_df['subject'] = "" # No separate subject
                
                # Try to extract Subject from body if it starts with "Subject:"
                # Regex: look for Subject: ... (up to newline)
                std_df['subject'] = std_df['body'].astype(str).str.extract(r'Subject:\s*(.*?)(?:\n|$)', flags=re.IGNORECASE).fillna("")
                
                # Try to extract Sender from body if it starts with "From:"
                std_df['sender'] = std_df['body'].astype(str).str.extract(r'From:\s*(.*?)(?:\n|$)', flags=re.IGNORECASE).fillna("")
                
                # If we extracted metadata from the body, we should try to clean the body to remove headers
                # This is optional but cleaner. For now we keep body as is to not lose context.
            else:
                print(f"Skipping {filename}: No 'body' or 'text_combined' column.")
                continue
                
            # Map Subject
            if 'subject' in df.columns:
                std_df['subject'] = df['subject']
            elif 'text_combined' not in df.columns:
                 std_df['subject'] = ""
                 
            # Map Other Fields (Sender, Receiver, Date, URLs)
            for field in ['sender', 'receiver', 'date', 'urls']:
                # Only overwrite if not already set (we might have inferred sender above)
                if field not in std_df.columns:
                    if field in df.columns:
                        std_df[field] = df[field]
                    else:
                        std_df[field] = ""
                    
            # Fill NaN with empty strings
            std_df = std_df.fillna("")
            
            # --- Attachment Inference Logic ---
            # We scan the body for attachment patterns and store them as a list (or empty list)
            def extract_attachments(text):
                if not isinstance(text, str):
                    return []
                return attachment_regex.findall(text)
            
            # Apply extraction (this might be slow on huge datasets, but we do it once per load)
            # Optimization: Only run on rows that contain "file" or "attach" to speed up
            # But for <200k rows it should be manageable in seconds
            
            # Using a simple apply here
            std_df['attachment_names'] = std_df['body'].apply(extract_attachments)
            
            # Check if we found any
            count_with_att = std_df[std_df['attachment_names'].map(len) > 0].shape[0]
            if count_with_att > 0:
                print(f"  Inferred attachments for {count_with_att} emails in {filename}")
            
            # Append to list
            all_dfs.append(std_df)
            print(f"Loaded {filename}: {len(std_df)} rows")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    if not all_dfs:
        raise ValueError("No datasets were loaded successfully.")
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Final standardization of labels just in case
    # Convert to numeric, coercing errors to NaN
    final_df['label'] = pd.to_numeric(final_df['label'], errors='coerce')
    # Drop rows where label could not be determined
    final_df = final_df.dropna(subset=['label'])
    final_df['label'] = final_df['label'].astype(int)
    
    # --- Quality Filter ---
    # Drop rows with extremely short bodies (likely parsing errors or useless data)
    # Threshold: 20 chars (e.g. "hello" or "test" might be valid spam, but "a" is not)
    initial_count = len(final_df)
    final_df = final_df[final_df['body'].fillna("").astype(str).str.len() > 20]
    filtered_count = len(final_df)
    
    if initial_count > filtered_count:
        print(f"Filtered {initial_count - filtered_count} rows due to insufficient content length (<20 chars).")
    
    print(f"Total unified records: {len(final_df)}")
    print(f"Class distribution:\n{final_df['label'].value_counts()}")
    
    return final_df

if __name__ == "__main__":
    # Test run
    # Assuming script is run from project root or evaluation dir
    # We need to find the root where CSVs are
    
    # If run from def-acc-logiq/evaluation/
    root_path = Path(__file__).parents[2] # Go up to hackathon-nov25
    
    print(f"Looking for datasets in: {root_path}")
    df = load_and_unify_datasets(str(root_path))
    print(df.head())
    
    # Show example with attachments
    print("\nExample with attachments:")
    print(df[df['attachment_names'].map(len) > 0].head(1)[['body', 'attachment_names']])
