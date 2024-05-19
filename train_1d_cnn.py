import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
from collections import Counter
import wandb
import os

# Load NLTK resources
import nltk
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


device = 'cpu'
if torch.cuda.is_available():
  device = 'cuda'

# Define preprocessing function
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)

    # print(tokens)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text


def remove_unfrequent_words(text):

  tokens = text.split()

  res  = []
  for t in tokens:
    if t in top_words:
      res.append(t)

  final_text = ' '.join(res)
  return final_text


data_train = pd.read_csv(r"text_data/train.csv" , encoding='latin1')

data_test = pd.read_csv(r"text_data/test.csv" , encoding='latin1')

data_train.dropna(axis=0, how='any', inplace=True)
data_test.dropna(axis=0, how='any', inplace=True)

print(data_train.columns)
print(data_test.columns)

# 0 is negative.
# 2 is neutral.
# 4 is positive .
unique_values = data_train['sentiment'].unique()
print(unique_values)
unique_values = data_test['sentiment'].unique()
print(unique_values)

# Define a dictionary to map sentiment values to their corresponding numbers
sentiment_mapping = {
    'negative': 0,
    'neutral': 2,
    'positive': 4
}

# for train data
# Replace sentiment values with their corresponding numbers
data_train['sentiment'] = data_train['sentiment'].replace(sentiment_mapping)
# Print the unique values in the 'sentiment' column after the replacement
print(data_train['sentiment'].unique())


# for test data
# Replace sentiment values with their corresponding numbers
data_test['sentiment'] = data_test['sentiment'].replace(sentiment_mapping)
# Print the unique values in the 'sentiment' column after the replacement
print(data_test['sentiment'].unique())


# Preprocess the text data
data_train['text'] = data_train['text'].apply(preprocess_text)
data_test['text'] = data_test['text'].apply(preprocess_text)

# Tokenize each sentence and count the frequency of words
word_counts = Counter(word_tokenize(' '.join(data_train['text'])))

# Convert the Counter object to a dataframe with word and frequency columns
word_counts_df = pd.DataFrame.from_dict(word_counts, orient='index', columns=['frequency'])

# Reset index to move the words from index to a column
word_counts_df.reset_index(inplace=True)

# Rename the columns
word_counts_df.columns = ['word', 'frequency']

# Sort the dataframe by frequency in descending order
word_counts_df = word_counts_df.sort_values(by='frequency', ascending=False)

# display(word_counts_df)

# Display the dataframe
top_word_counts_df = word_counts_df[:top_k_words]
# display(top_word_counts_df)

print(f'top {top_k_words} words are selected')
top_words = top_word_counts_df['word'].to_list()
print(top_words)
print(len(top_words))

# import pickle

# # Save the top_words list to disk using pickle
# with open('top_words.pkl', 'wb') as f:
#     pickle.dump(top_words, f)


# Preprocess the text data
data_train['text'] = data_train['text'].apply(remove_unfrequent_words)

data_test['text'] = data_test['text'].apply(remove_unfrequent_words)

# Vectorize the text data
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(data_train['text'])
y_train = data_train['sentiment']

X_test = vectorizer.transform(data_test['text'])
y_test = data_test['sentiment']

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)

X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.long)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

torch.device(device)

# Move tensors to the GPU
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Define a custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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


# Hyperparameters
vocab_size = len(top_words)
embedding_dim = cnn1d_config['embedding_dim']
output_dim = cnn1d_config['output_dim']
kernel_size = cnn1d_config['kernel_size']
stride = cnn1d_config['stride']
num_epochs = cnn1d_config['num_epochs']
batch_size = cnn1d_config['batch_size']
learning_rate = cnn1d_config['lr']

API_KEY = cnn1d_config['API_KEY']

os.environ["WANDB_API_KEY"] = API_KEY

# Initialize the model
kernel_sizes = [3, 4, 5]
model = CNN1D(vocab_size, embedding_dim, output_dim, kernel_sizes, stride)

model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create datasets and dataloaders
train_dataset = SentimentDataset(X_train, y_train)
test_dataset = SentimentDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)



# Initialize wandb
wandb.init(project="text_analysis", name="1d_cnn_metrics1")

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.long())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    # Evaluation on test set
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.long())
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    test_accuracy = 100 * test_correct / test_total

    # Log metrics with wandb
    wandb.log({
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy
    })

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Test Accuracy: {test_accuracy:.2f}%')

    model.train()

# Save model after training
torch.save(model.state_dict(), "models/1d_cnn.pth")
wandb.save()
