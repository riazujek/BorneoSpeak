import os
import pandas as pd
from datasets import Dataset
from transformers import MarianMTModel, MarianTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "data", "final", "synthetic_parallel_corpus_CLEANED.csv")
model_output_dir = os.path.join(project_root, "models", "iban_nmt_v1")

# 1. Load Parent Model and BPE Tokenizer
model_checkpoint = "Helsinki-NLP/opus-mt-ms-en"
tokenizer = MarianTokenizer.from_pretrained(model_checkpoint)
model = MarianMTModel.from_pretrained(model_checkpoint)

# 2. Load and Prepare Dataset
df = pd.read_csv(data_path)
df = df.dropna()

# Convert pandas dataframe to HuggingFace Dataset
raw_dataset = Dataset.from_pandas(df)
raw_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42) # 80/20 split for training and BLEU evaluation

# 3. Tokenization Function
def preprocess_function(examples):
    inputs = [str(ex) for ex in examples["iban_sentence"]]
    targets = [str(ex) for ex in examples["english_sentence"]]
    
    # Tokenize source and target sequences with a 128 max length constraint
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)
        
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = raw_dataset.map(preprocess_function, batched=True)

# 4. Define Training Hyperparameters (From Methodology)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True, # Enables mixed precision for faster GPU training
)

# 5. Initialize Trainer and Begin Fine-Tuning
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print(f"Starting NMT training on {len(df)} sentence pairs...")
trainer.train()

# 6. Save the Localized AI Brain
trainer.save_model(model_output_dir)
print(f"Training complete. Offline model saved to {model_output_dir}")