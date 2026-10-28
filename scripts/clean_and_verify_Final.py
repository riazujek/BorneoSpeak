import pandas as pd
import numpy as np
import os

# Dynamically route to your project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Absolute paths
input_file = os.path.join(project_root, "data", "final", "synthetic_parallel_corpus.csv")
output_file = os.path.join(project_root, "data", "final", "synthetic_parallel_corpus_CLEANED.csv")
qa_sample_file = os.path.join(project_root, "data", "final", "QA_sample.csv")

print(f"1. Loading raw corpus from: {input_file}")
df = pd.read_csv(input_file)
original_count = len(df)

# Ensure our core columns exist
if 'iban_sentence' not in df.columns:
    df['iban_sentence'] = np.nan
if 'english_sentence' not in df.columns:
    df['english_sentence'] = np.nan

# --- STEP 1: SALVAGE MISPLACED SENTENCES ---
print("2. Salvaging misplaced Iban sentences from hallucinated columns...")
def salvage_iban(row):
    if pd.notna(row['iban_sentence']) and str(row['iban_sentence']).strip() != "":
        return str(row['iban_sentence']).strip()
    
    for col in df.columns:
        if col not in ['iban_sentence', 'english_sentence']:
            val = row[col]
            if pd.notna(val) and str(val).strip() != "":
                return str(val).strip()
    return np.nan

df['iban_sentence'] = df.apply(salvage_iban, axis=1)

# --- STEP 2: DROP JUNK COLUMNS ---
print("3. Dropping hallucinated headers...")
df = df[['iban_sentence', 'english_sentence']]

# --- STEP 3: CLEAN BLANKS & DUPLICATES ---
print("4. Running basic sanitization...")
df = df.dropna(subset=['iban_sentence', 'english_sentence'])
df = df[df['iban_sentence'].str.strip() != '']
df = df[df['english_sentence'].str.strip() != '']
df = df.drop_duplicates()

# --- STEP 4: LENGTH RATIO QA CHECK ---
print("5. Running QA: Length Ratio Check...")
def check_length_ratio(row):
    iban_words = len(str(row['iban_sentence']).split())
    eng_words = len(str(row['english_sentence']).split())
    
    if iban_words == 0 or eng_words == 0:
        return False
        
    ratio = eng_words / iban_words
    return 0.3 <= ratio <= 3.5

df = df[df.apply(check_length_ratio, axis=1)]

df.to_csv(output_file, index=False, encoding='utf-8')
final_count = len(df)

# --- STEP 5: GENERATE HUMAN REVIEW SAMPLE ---
print("6. Generating 100-word random sample for human review...")
if len(df) > 100:
    qa_sample = df.sample(n=100, random_state=42)
else:
    qa_sample = df

qa_sample.to_csv(qa_sample_file, index=False, encoding='utf-8')

print("\n=== PIPELINE COMPLETE ===")
print(f"Original Rows: {original_count}")
print(f"Cleaned Rows:  {final_count}")
print(f"Rows Dropped:  {original_count - final_count}")