import os
import json
import time
import pandas as pd
import google.generativeai as genai

# 1. Setup & Configuration
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "nothin to see here"))
model = genai.GenerativeModel('gemini-3.1-flash-lite') 

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

input_file = os.path.join(project_root, "data", "final", "cleaned_iban_dataset_MASTER.csv")
output_file = os.path.join(project_root, "data", "final", "synthetic_parallel_corpus.csv")

# 2. Load Master Dictionary & Handle Resuming
df = pd.read_csv(input_file)
words_to_process = df.to_dict('records')

generated_pairs = []
start_index = 0

# If the output file already exists, load it and skip the words we already did
if os.path.exists(output_file):
    try:
        existing_df = pd.read_csv(output_file)
        generated_pairs = existing_df.to_dict('records')
        start_index = len(generated_pairs)
        print(f"Found existing corpus. Resuming from word {start_index}...")
    except Exception as e:
        print("Could not load existing file, starting fresh.")

# 3. Batching Logic
BATCH_SIZE = 10
words_left = words_to_process[start_index:]

print(f"Starting synthetic generation for the remaining {len(words_left)} words...")

for i in range(0, len(words_left), BATCH_SIZE):
    batch = words_left[i:i + BATCH_SIZE]
    current_batch_num = (start_index + i) // BATCH_SIZE + 1
    
    prompt = """
    You are an expert Iban linguist. I will give you a list of Iban words and their English definitions.
    For each word, write ONE grammatically perfect, natural everyday Iban sentence using the word, and provide its exact English translation.
    
    Return the result STRICTLY as a valid JSON array of objects.
    Each object MUST have exactly two keys: "iban_sentence" and "english_sentence".
    Ensure every object is separated by a comma. Escape any internal double quotes using a backslash.
    Do not include any extra text outside the JSON array. Do not use markdown code blocks.
    
    WORDS:
    """
    
    for item in batch:
        prompt += f"- Word: {item['source_word']} | Definition: {item['english_translation']}\n"
        
    # Auto-Retry Loop (Tries up to 3 times if Gemini formats JSON badly)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            raw_text = response.text.strip()
            
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            json_data = json.loads(raw_text.strip())
            generated_pairs.extend(json_data)
            
            print(f"Successfully processed batch {current_batch_num} (Generated {len(generated_pairs)} total sentences)")
            
            # Save progressively
            pd.DataFrame(generated_pairs).to_csv(output_file, index=False, encoding='utf-8')
            
            time.sleep(5) # Pace the API
            break # Break out of the retry loop if successful
            
        except Exception as e:
            print(f"\n[!] JSON Parsing Error on batch {current_batch_num}, Attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print("[!] Max retries hit for this batch. Skipping batch to keep going...")
            else:
                print("Retrying in 5 seconds...")
                time.sleep(5)

print(f"\nGeneration run finished! Total pure sentence pairs saved: {len(generated_pairs)}")