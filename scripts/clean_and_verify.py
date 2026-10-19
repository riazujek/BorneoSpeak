import os
import re
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
# Define both input files
input_file_1 = os.path.join(script_dir, "ai_parsed_iban_dataset.csv")
input_file_2 = os.path.join(script_dir, "ai_parsed_iban_dataset_PART2.csv")
output_file = os.path.join(script_dir, "cleaned_iban_dataset_MASTER.csv")

# Load and combine both files
print("--- Loading and Merging Datasets ---")
try:
    df1 = pd.read_csv(input_file_1)
    df2 = pd.read_csv(input_file_2)
    df = pd.concat([df1, df2], ignore_index=True)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

print(f"Initial raw rows combined: {len(df)}")

def repair_mojibake_and_clean(text):
    if not isinstance(text, str):
        return ""
    
    # Replace Windows-1252 / UTF-8 Mojibake representations using explicit unicode escapes
    replacements = {
        "\u00e2\u20ac\u2122": "'",
        "\u00e2\u20ac\u02dc": "'",
        "\u00e2\u20ac\u0153": '"',
        "\u00e2\u20ac\u009d": '"',
        "\u00e2\u20ac\u2013": "-",
        "\u00e2\u20ac\u2014": "-",
        "\u00c2": "",
        "’": "'",
        "‘": "'",
        "`": "'"
    }
    
    for bad_seq, good_char in replacements.items():
        text = text.replace(bad_seq, good_char)
    
    # Collapse multiple whitespace characters into a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
df['source_word'] = df['source_word'].apply(repair_mojibake_and_clean)
df['english_translation'] = df['english_translation'].apply(repair_mojibake_and_clean)

# Drop missing values and short noise
df = df.dropna(subset=['source_word', 'english_translation'])
df = df[(df['source_word'].str.len() > 1) & (df['english_translation'].str.len() > 1)]

# Deduplicate headwords (crucial for removing overlap at the "tugang" boundary)
df = df.drop_duplicates(subset=['source_word'])

# Drop rows where an English stopword appears as the Iban headword (column inversion check)
english_stopwords = {"the", "a", "an", "is", "are", "to", "and", "or", "of", "in", "on", "for"}
df = df[~df['source_word'].str.lower().isin(english_stopwords)]

# Save output
df.to_csv(output_file, index=False, encoding='utf-8')

print("\nSanitization and Merge Successful!")
print(f"Cleaned master rows saved: {len(df)}")
print(f"Saved to: {output_file}")