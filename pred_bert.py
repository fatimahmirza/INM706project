import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification

import yaml

# Load YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access variables for each model
bert_config = config['models']['BERT']


# Set device
device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'

# Load the saved BERT model using torch.load
model_path = 'models/bert_best.pth'
model_state_dict = torch.load(model_path ,map_location=torch.device(device))

# Initialize a new instance of BertForSequenceClassification with the correct number of labels
num_labels = 3  # Assuming your model has three output classes
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

# Load the saved model state_dict into the new model
model.load_state_dict(model_state_dict)

# Ensure the model is in evaluation mode
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


# Function to perform inference on a single sentence
def predict_sentiment(sentence):
    inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class

# Test the 10 sentences
test_sentences = [
    "i have nothing to say",
    "I absolutely loved the movie, it was fantastic!",
    "The food at that restaurant was terrible, I wouldn't recommend it to anyone.",
    "Today is a beautiful day, perfect for a picnic in the park.",
    "The customer service I received was excellent, they were very helpful and polite.",
    "I'm feeling really disappointed with the service I got from that company.",
    "The new album from my favorite band is amazing, I can't stop listening to it.",
    "The traffic this morning was awful, it took me twice as long to get to work.",
    "I'm indifferent about the new policy changes, they don't really affect me.",
    "The weather forecast predicts rain for the entire weekend, what a downer.",
    "The book I read yesterday was just okay, nothing special."
]

from tabulate import tabulate

# Initialize an empty list to store the data
data = []

# Iterate through the test sentences
for sentence in test_sentences:
    # Get the predicted label for the current sentence
    predicted_label = predict_sentiment(sentence)

    # Determine the sentiment label based on the predicted label
    sentiment = "Negative" if predicted_label == 0 else \
                "Positive" if predicted_label == 2 else \
                "Neutral"

    # Append the data to the list
    data.append([sentence, sentiment])

# Print the predictions using tabulate
print(tabulate(data, headers=["Test Sentence", "Predicted Sentiment"]))
