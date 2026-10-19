import os
import json
import time
import pandas as pd
from google import genai

# Initialize the new SDK client
client = genai.Client(api_key="nothing to see here")

def extract_with_ai(chunk):
    prompt = f"""
    You are a linguistic data extractor. I am giving you a messy, OCR-scanned section of an Iban-to-English dictionary.
    Extract ONLY the Iban root words and their English definitions. 
    Return the result STRICTLY as a valid JSON array of objects with the keys "source_word" and "english_translation".
    Do NOT wrap the response in markdown code blocks like ```json.
    
    TEXT:
    {chunk}
    """
    try:
        # Use the newest, fastest, and most widely supported free-tier model
        response = client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt
        )

        # Failsafe: Clean up potential markdown wrappers just in case the AI adds them
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
    except Exception as e:
        print(f"Error parsing chunk: {e}")
        return []

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, "An Iban-English Dictionary by Anthony Richards_djvu - Copy.txt")
    output_file = os.path.join(script_dir, "ai_parsed_iban_dataset_PART2.csv") # Save as a new file so we don't overwrite A-T
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        full_text = f.read()

    # Find where "tugang" is in the raw text to resume extraction
    resume_index = full_text.find("tugang")
    
    if resume_index != -1:
        print(f"Found 'tugang' at character index {resume_index}. Resuming from here...")
        # Slice the text to only include everything AFTER "tugang"
        full_text = full_text[resume_index:]
    else:
        print("Could not find 'tugang'. Starting from the beginning.")

    chunk_size = 5000
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    
    final_dataset = []
    print(f"Starting AI extraction on {len(chunks)} chunks using gemini-3.5-flash-lite...")

    for idx, chunk in enumerate(chunks):
        print(f"Processing chunk {idx + 1}/{len(chunks)}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            parsed_data = extract_with_ai(chunk)
            
            if isinstance(parsed_data, list) and len(parsed_data) > 0:
                final_dataset.extend(parsed_data)
                
                # Progressively save to avoid losing data if it crashes again
                temp_df = pd.DataFrame(final_dataset).drop_duplicates(subset=['source_word'])
                temp_df.to_csv(output_file, index=False, encoding='utf-8')
                break 
            else:
                print(f"Chunk {idx + 1} failed. Retrying in 10 seconds... (Attempt {attempt + 1} of {max_retries})")
                time.sleep(10)
                
        time.sleep(2) 
        
    print(f"\nExtraction complete! Saved to {output_file}.")