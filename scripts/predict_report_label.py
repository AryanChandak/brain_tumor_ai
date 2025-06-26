from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "models/report_bert"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Put model in evaluation mode
model.eval()

# ID-to-label mapping from saved model
id2label = model.config.id2label

# Sample input
text = input("Enter MRI report text: ")

# Tokenize input
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predicted_label_id = torch.argmax(outputs.logits, dim=1).item()
    predicted_label = id2label[predicted_label_id]

print(f"🧠 Predicted Tumor Type: {predicted_label}")
