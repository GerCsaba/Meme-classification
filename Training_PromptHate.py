import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset

# Load the data
negative_file = 'Final Dataset_Negativ_Train - 50%.csv'
positive_file = 'Final Dataset_Positive_Train - 50%.csv'

data_neg = pd.read_csv(negative_file)
data_pos = pd.read_csv(positive_file)

# Extract relevant columns
data_neg['label'] = 0  # Negative label
data_pos['label'] = 1  # Positive label
# print(data_neg.columns)
# print(data_pos.columns)
data = pd.concat([data_neg[['Description', 'Extracted Text', 'label']], data_pos[['Description', 'Extracted Text', 'label']]], ignore_index=True)
data.columns = ['description', 'text', 'label']

# Combine description and text as input
data['input'] = data['description'] + ' ' + data['text']

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['input'], data['label'], test_size=0.2, random_state=42
)

# Load RoBERTa tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=512)

train_encodings = tokenize_function(train_texts.tolist())
val_encodings = tokenize_function(val_texts.tolist())

# Convert to Hugging Face Dataset format
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels.tolist()
})
val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels.tolist()
})

# Load RoBERTa model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",  # Ez egyezik a save_strategy-vel
    save_strategy="steps",
    save_steps=500,  # Mentés minden 500 lépés után
    eval_steps=500,  # Kiértékelés minden 500 lépés után
    load_best_model_at_end=True,
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs_v3_50%",
    logging_steps=10,
)


# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./roberta_meme_modelv3-50%_6epoch')
tokenizer.save_pretrained('./roberta_meme_modelv3-50%_6epoch')

print("Model training complete and saved.")
