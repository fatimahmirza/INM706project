import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os 

import yaml

# Load YAML file
with open('config.yaml', 'r') as file:
  config = yaml.safe_load(file)
  
bert_config = config['models']['BERT']

API_KEY = bert_config['API_KEY']
os.environ["WANDB_API_KEY"] = API_KEY

# Load data
train_df = pd.read_csv("text_data/train.csv" ,  encoding ='latin1')
test_df = pd.read_csv("text_data/test.csv" ,   encoding ='latin1')

train_df.dropna(axis=0, how='any', inplace=True)
test_df.dropna(axis=0, how='any', inplace=True)

# Preprocess data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

# Tokenize text
max_length = bert_config['max_length']
train_inputs = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
test_inputs = tokenizer(test_df['text'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

# Encode labels
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_df['sentiment'])
test_labels = label_encoder.transform(test_df['sentiment'])

# Create TensorDatasets
train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], torch.tensor(train_labels, dtype=torch.long))
test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], torch.tensor(test_labels, dtype=torch.long))

# Define data loaders
batch_size = bert_config['batch_size']
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)


import wandb

# Initialize wandb
wandb.init(project="text_analysis", name="bert_metrics1")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=bert_config['lr'], eps=1e-8)
epochs = 3
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

from tqdm import tqdm

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_correct_train = 0
    total_predictions_train = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False)
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        # Calculate training accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total_correct_train += torch.sum(predictions == batch[2]).item()
        total_predictions_train += batch[2].size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({'Loss': loss.item()})

    avg_train_loss = total_loss / len(train_dataloader)
    train_accuracy = total_correct_train / total_predictions_train

    wandb.log({"train_accuracy": train_accuracy})

    print(f"Epoch {epoch+1}/{epochs} - Average Training Loss: {avg_train_loss:.4f} - Training Accuracy: {train_accuracy:.4f}")

    # Evaluation loop
    model.eval()
    total_correct_test = 0
    total_predictions_test = 0

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
            outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        total_correct_test += torch.sum(predictions == batch[2]).item()
        total_predictions_test += batch[2].size(0)

    test_accuracy = total_correct_test / total_predictions_test

    wandb.log({"test_accuracy": test_accuracy})

    print(f"Epoch {epoch+1}/{epochs} - Test Accuracy: {test_accuracy:.4f}")

    model.train()  # Switch back to training mode

# Save the trained model
torch.save(model.state_dict(), 'models/bert.pth')
wandb.finish()  # Finish logging