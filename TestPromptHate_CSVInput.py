import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# Load the trained model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('./roberta_meme_modelv3-50%_6epoch')
model = RobertaForSequenceClassification.from_pretrained('./roberta_meme_modelv3-50%_6epoch')

# Function to predict sentiment of meme
def predict_sentiment(description, text):
    # Combine description and text
    input_text = description + ' ' + text
    
    # Tokenize the input text
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    # Get the model's prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits / 2, dim=-1)
    
    # Get the predicted label (0 for Negative, 1 for Positive)
    predicted_label = torch.argmax(probabilities, dim=-1).item()
    return predicted_label

# Load the positive and negative test datasets
positive_test_file = 'Dataset_Positive_Test.csv'
negative_test_file = 'Dataset_Negativ_Test.csv'

# Read the datasets
positive_test_data = pd.read_csv(positive_test_file)
negative_test_data = pd.read_csv(negative_test_file)

# We assume 'Description' and 'Extracted Text' are the columns we want to use
positive_test_texts = positive_test_data[['Description', 'Extracted Text']].dropna()
negative_test_texts = negative_test_data[['Description', 'Extracted Text']].dropna()

# Initialize counters
correct_predictions = 0
total_predictions = 0

# Test the model with positive examples
for idx, row in positive_test_texts.iterrows():
    description = row['Description']
    text = row['Extracted Text']
    predicted_label = predict_sentiment(description, text)
    
    # True label is 1 (Positive)
    if predicted_label == 1:
        correct_predictions += 1
    total_predictions += 1

# Test the model with negative examples
for idx, row in negative_test_texts.iterrows():
    description = row['Description']
    text = row['Extracted Text']
    predicted_label = predict_sentiment(description, text)
    
    # True label is 0 (Negative)
    if predicted_label == 0:
        correct_predictions += 1
    total_predictions += 1

# Calculate and print accuracy
if total_predictions > 0:
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"\nAccuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions})")
else:
    print("\nNincsenek feldolgozható tesztadatok.")
