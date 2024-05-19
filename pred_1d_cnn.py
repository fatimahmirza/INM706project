import torch
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import yaml
from torchtext.data.utils import get_tokenizer
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
from tabulate import tabulate
import torch.nn as nn
import pickle


device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'

class CNN1D(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, kernel_sizes, stride):
        super(CNN1D, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=output_dim, kernel_size=ks, stride=stride)
            for ks in kernel_sizes
        ])
        self.relu = nn.ReLU()
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(output_dim) for _ in range(len(kernel_sizes))])
        self.fc = nn.Linear(len(kernel_sizes) * output_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1)  # Change dimensions for Conv1D
        conv_outs = [self.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv_out, conv_out.shape[2]).squeeze(2) for conv_out in conv_outs]
        cat = torch.cat(pooled, dim=1)
        output = self.fc(cat)
        return output


# Load NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Access variables for each model
cnn1d_config = config['models']['1D_CNN']

# put -1 to consider all words of vocab
top_k_words = cnn1d_config['vocab_size']

# Define preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

# Define function to remove infrequent words
def remove_unfrequent_words(text):
    tokens = text.split()
    res = []
    for t in tokens:
        if t in top_words:
            res.append(t)
    final_text = ' '.join(res)
    return final_text

# Load the test sentences
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

# Preprocess the test sentences
preprocessed_test_sentences = [preprocess_text(sentence) for sentence in test_sentences]

# Tokenization function
tokenizer = get_tokenizer("basic_english")

# Tokenize the preprocessed test sentences
tokenized_test_sentences = [tokenizer(sentence) for sentence in preprocessed_test_sentences]


# Load the top_words list from disk
with open('top_words.pkl', 'rb') as f:
    top_words = pickle.load(f)

# Convert tokens to numerical format (TF-IDF vectors)
vectorizer = TfidfVectorizer(vocabulary=top_words)
X_test = vectorizer.fit_transform(preprocessed_test_sentences)

# Convert to PyTorch tensor
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
X_test = X_test.to(device, dtype=torch.long)
# Hyperparameters
vocab_size = len(top_words)
embedding_dim = cnn1d_config['embedding_dim']
output_dim = cnn1d_config['output_dim']
kernel_size = cnn1d_config['kernel_size']
stride = cnn1d_config['stride']
num_epochs = cnn1d_config['num_epochs']
batch_size = cnn1d_config['batch_size']
learning_rate = cnn1d_config['lr']

# Initialize the model
kernel_sizes = [3, 4, 5]
model = CNN1D(vocab_size, embedding_dim, output_dim, kernel_sizes, stride)
model.load_state_dict(torch.load("models/1d_cnn_best.pth"))
model.to(device)
model.eval()

# Perform inference on the test sentences
with torch.no_grad():
    # Move test data to the appropriate device
    X_test = X_test.to(device)
    # Perform inference
    outputs = model(X_test)
    # Get predicted labels
    _, predicted_labels = torch.max(outputs, 1)

# Map numerical labels back to their original sentiment values
label_mapping = {0: 'negative', 2: 'neutral', 4: 'positive'}
predicted_sentiments = [label_mapping[label.item()] for label in predicted_labels]


# Create a list to hold the results
results = []

# Iterate through the test sentences and predicted sentiments
for sentence, sentiment in zip(test_sentences, predicted_sentiments):
    results.append([sentence, sentiment])

print()
# Print the results using tabulate
print(tabulate(results, headers=["Sentence", "Predicted Sentiment"]))
print()
