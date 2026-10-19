import os
import json
import time
import pandas as pd
import google.generativeai as genai

# 1. Setup & Configuration
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "nothing to see here"))
model = genai.GenerativeModel('gemini-3.5-flash-lite')

script_dir = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.join(script_dir, "cleaned_iban_dataset.csv")
output_file = os.path.join(script_dir, "synthetic_parallel_corpus.csv")

# 2. Load Cleaned Dictionary
df = pd.read_csv(input_file)
words_to_process = df.to_dict('records')

# 3. Batching & Extraction Logic
BATCH_SIZE = 10
generated_pairs = []

print(f"Starting synthetic generation for {len(words_to_process)} words...")

for i in range(0, len(words_to_process), BATCH_SIZE):
    batch = words_to_process[i:i + BATCH_SIZE]
    
    # Construct the prompt
    prompt = "You are an expert Iban linguist. I will give you a list of Iban words and their English definitions. "
    prompt += "For each word, write one grammatically perfect, natural Iban sentence using the word, and provide its exact English translation. "
    prompt += "Return the result STRICTLY as a JSON array of objects with keys 'iban_sentence' and 'english_sentence'. Do not include markdown formatting like ```json.\n\n"
    
    for item in batch:
        prompt += f"- Word: {item['source_word']} | Definition: {item['english_translation']}\n"
        
    try:
        response = model.generate_content(prompt)
        text_output = response.text.strip().removeprefix("```json").removesuffix("```").strip()
        
        # Parse JSON and append to our dataset
        json_data = json.loads(text_output)
        generated_pairs.extend(json_data)
        
        print(f"Successfully processed batch {i // BATCH_SIZE + 1} (Generated {len(generated_pairs)} sentences so far)")
        
        # Save progressively to protect data against rate limits or crashes
        temp_df = pd.DataFrame(generated_pairs)
        temp_df.to_csv(output_file, index=False, encoding='utf-8')
        
        # Brief pause to respect API rate limits
        time.sleep(2)
        
    except Exception as e:
        print(f"Error on batch {i // BATCH_SIZE + 1}: {e}")
        print("Saving current progress and pausing...")
        break

print(f"\nGeneration complete or halted! Total pure sentence pairs saved: {len(generated_pairs)}")
print(f"File saved to: {output_file}")