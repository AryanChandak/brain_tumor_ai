import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

# Load dataset
df = pd.read_csv("data/brain_reports.csv")

# Encode labels
label2id = {label: i for i, label in enumerate(df["label"].unique())}
id2label = {v: k for k, v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

# Split dataset
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df[["report_text", "label_id"]])
val_ds = Dataset.from_pandas(val_df[["report_text", "label_id"]])

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["report_text"], truncation=True, padding=True)

train_ds = train_ds.map(tokenize, batched=True)
train_ds = train_ds.rename_column("label_id", "labels")
train_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

val_ds = val_ds.map(tokenize, batched=True)
val_ds = val_ds.rename_column("label_id", "labels")
val_ds.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])


# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training arguments
training_args = TrainingArguments(
    output_dir="models/report_bert",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1
)

# Define evaluation metric
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, classification_report
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), axis=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train
trainer.train()
trainer.save_model("models/report_bert")
print("✅ BERT model saved to: models/report_bert")
