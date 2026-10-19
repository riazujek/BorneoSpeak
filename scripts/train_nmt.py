import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, "cleaned_iban_dataset.csv")

# 1. Load Sanitized Data
df = pd.read_csv(dataset_path)
df = df.rename(columns={"source_word": "id", "english_translation": "en"})

# Convert to Hugging Face Dataset and Train/Val split (90/10)
raw_dataset = Dataset.from_pandas(df)
dataset_split = raw_dataset.train_test_split(test_size=0.1, seed=42)

# 2. Base Model & Tokenizer
model_name = "Helsinki-NLP/opus-mt-id-en"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Preprocessing
def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["id"]]
    targets = [str(ex) for ex in examples["en"]]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(text_target=targets, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset_split.map(preprocess_function, batched=True)

# 4. Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./iban_nmt_checkpoints",
    eval_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_steps=100,
    save_strategy="epoch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 5. Trainer Initialization (Updated for latest Hugging Face API)
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

# 6. Execute Training & Save Final Artifact
trainer.train()

final_save_path = os.path.join(script_dir, "iban_translator_model")
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print(f"Fine-tuning complete. Artifact saved to: {final_save_path}")