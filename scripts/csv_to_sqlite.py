import sqlite3
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, "cleaned_iban_dataset_MASTER.csv")
db_file = os.path.join(script_dir, "iban_dictionary_MASTER.db")

print("Reading CSV data...")
df = pd.read_csv(csv_file)

# Rename columns to be clean and app-friendly
df.rename(columns={'source_word': 'iban_word', 'english_translation': 'english_definition'}, inplace=True)

print("Connecting to SQLite database...")
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Write the data to a new table named 'dictionary'
df.to_sql('dictionary', conn, if_exists='replace', index=False)

# Create an index on the Iban word column for zero-latency search in Flutter
print("Building search indexes...")
cursor.execute('CREATE INDEX idx_iban_word ON dictionary(iban_word)')

conn.commit()
conn.close()

print(f"Success! Database created at: {db_file}")