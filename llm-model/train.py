import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

# Load Excel dataset
df = pd.read_excel("dataset/dataset.xlsx")

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

# Preprocess data
def preprocess(batch):
    inputs = tokenizer(batch["text"], max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(batch["summary"], max_length=50, truncation=True, padding="max_length")
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True, remove_columns=["id", "text", "summary"])

# Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
