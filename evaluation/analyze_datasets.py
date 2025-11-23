import pandas as pd
import glob
import os
import re
from pathlib import Path

def analyze_datasets(base_path: str):
    print(f"Analyzing CSV files in: {base_path}\n")
    
    search_pattern = os.path.join(base_path, "*.csv")
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print("No CSV files found.")
        return

    total_emails = 0
    total_with_attachments_potential = 0
    
    # Regex for potential attachment patterns in body
    # E.g. "attached file : name.ext", "filename=name.ext", "begin 644 name.ext"
    attachment_regex = re.compile(r'(?:attached file\s*[:\-]?\s*|filename\s*[=:]\s*|name\s*[=:]\s*"?)([\w\-. ]+\.[a-zA-Z0-9]{2,4})(?:"|\s|$)', re.IGNORECASE)
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"=== {filename} ===")
        
        try:
            # Try reading with different encodings
            try:
                df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin1', on_bad_lines='skip')
                
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            
            # Determine Body Column first
            body_col = None
            if 'body' in df.columns:
                body_col = 'body'
            elif 'text_combined' in df.columns:
                body_col = 'text_combined'

            # --- Quality Assessment ---
            if body_col:
                # 1. Duplicates
                num_duplicates = df[body_col].duplicated().sum()
                duplicate_pct = (num_duplicates / len(df)) * 100
                
                # 2. Content Length
                df['body_len'] = df[body_col].fillna("").astype(str).apply(len)
                avg_len = df['body_len'].mean()
                min_len = df['body_len'].min()
                # Count "garbage" (very short emails < 10 chars)
                garbage_count = df[df['body_len'] < 10].shape[0]
                garbage_pct = (garbage_count / len(df)) * 100
                
                # 3. Empty/Null
                empty_count = df[body_col].isna().sum() + (df[body_col].fillna("").astype(str).str.strip() == "").sum()
                empty_pct = (empty_count / len(df)) * 100
                
                print(f"  Quality Metrics:")
                print(f"    - Avg Body Length: {avg_len:.0f} chars")
                print(f"    - Duplicates:      {num_duplicates} ({duplicate_pct:.1f}%)")
                print(f"    - Empty/Null:      {empty_count} ({empty_pct:.1f}%)")
                print(f"    - Too Short (<10): {garbage_count} ({garbage_pct:.1f}%)")
            
            # Check Label Distribution
            # Normalize column names first
            df.columns = [c.lower().strip() for c in df.columns]
            
            if 'label' in df.columns:
                print(f"  Label Distribution:\n{df['label'].value_counts().head()}")
            else:
                print("  [WARNING] No 'label' column found!")
                
            # Check for Attachment Columns
            att_cols = [c for c in df.columns if 'attach' in c or 'file' in c]
            if att_cols:
                print(f"  Potential Attachment Columns: {att_cols}")
                
            # Sample body scan for attachments
            if body_col:
                sample_size = min(1000, len(df))
                sample = df[body_col].dropna().astype(str).sample(sample_size)
                matches = sample.apply(lambda x: attachment_regex.findall(x))
                found_count = matches[matches.apply(len) > 0].count()
                
                print(f"  Regex Scan (Sample {sample_size}): Found potential attachments in {found_count} emails.")
                if found_count > 0:
                    print(f"  Example matches: {matches[matches.apply(len) > 0].head(3).tolist()}")
                    
                if found_count > 0:
                    total_with_attachments_potential += (found_count / sample_size) * len(df)
            
            total_emails += len(df)
            print("-" * 40)
            
        except Exception as e:
            print(f"  Error analyzing {filename}: {e}")
            print("-" * 40)
            
    print("\n=== Summary ===")
    print(f"Total Emails Scanned: {total_emails}")
    print(f"Estimated Emails with Attachments (via Regex): ~{int(total_with_attachments_potential)}")

if __name__ == "__main__":
    # Run from project root
    root_path = Path(__file__).parents[2]
    analyze_datasets(str(root_path))

