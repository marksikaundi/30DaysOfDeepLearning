Let's use the IMDb movie reviews dataset available from the TensorFlow Datasets library. This dataset is widely used for sentiment analysis and is readily accessible.

### Step 1: Setting Up the Environment

First, let's update the `requirements.txt` file to include TensorFlow Datasets:

```txt
nltk
scikit-learn
pandas
tensorflow
tensorflow-datasets
```

Install the dependencies using pip:

```sh
pip install -r requirements.txt
```

### Step 2: Data Preprocessing

Update the `src/data_preprocessing.py` file to load the IMDb dataset from TensorFlow Datasets:

```python
# src/data_preprocessing.py

import tensorflow_datasets as tfds
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('stopwords')

def load_data():
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_data, test_data = dataset['train'], dataset['test']
    return train_data, test_data

def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Removing punctuation and stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return ' '.join(tokens)

def prepare_data(data):
    texts = []
    labels = []
    for text, label in data:
        texts.append(preprocess_text(text.numpy().decode('utf-8')))
        labels.append(label.numpy())
    return texts, labels

def tokenize_and_pad_sequences(X_train, X_test, max_len=100):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    return X_train_pad, X_test_pad, tokenizer
```

### Step 3: Model Training

The `src/model_training.py` file remains the same:

```python
# src/model_training.py

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

def build_model(vocab_size, embedding_dim=100, max_len=100):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
        LSTM(128, return_sequences=True),
        Dropout(0.5),
        LSTM(64),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    return history
```

### Step 4: Model Evaluation

The `src/model_evaluation.py` file remains the same:

```python
# src/model_evaluation.py

from sklearn.metrics import classification_report, accuracy_score

def evaluate_model(model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    return accuracy, report
```

### Step 5: Main Script

Update the `main.py` file to use the new data loading function:

```python
# main.py

from src.data_preprocessing import load_data, prepare_data, tokenize_and_pad_sequences
from src.model_training import build_model, train_model
from src.model_evaluation import evaluate_model

# Load and preprocess data
train_data, test_data = load_data()
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# Tokenize and pad sequences
X_train_pad, X_test_pad, tokenizer = tokenize_and_pad_sequences(X_train, X_test)
vocab_size = len(tokenizer.word_index) + 1

# Build and train model
model = build_model(vocab_size)
history = train_model(model, X_train_pad, y_train, X_test_pad, y_test)

# Evaluate model
accuracy, report = evaluate_model(model, X_test_pad, y_test)
print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
```

### Step 6: README

Update the `README.md` file to reflect the use of TensorFlow Datasets:

```markdown
# NLP Project: Sentiment Analysis on IMDb Reviews

## Overview

This project demonstrates a complete NLP pipeline for sentiment analysis on IMDb movie reviews. The goal is to classify reviews as positive or negative.

## Project Structure
```

nlp_project/
│
├── data/
│ ├── (No need for local data files as we are using TensorFlow Datasets)
│
├── src/
│ ├── data_preprocessing.py
│ ├── model_training.py
│ ├── model_evaluation.py
│
├── main.py
│
├── requirements.txt
│
└── README.md

````

## Setup
1. Install dependencies:
   ```sh
   pip install -r requirements.txt
````

2. Run the main script:
   ```sh
   python main.py
   ```

## Results

The model's accuracy and classification report will be printed in the console.

```

### Summary

This updated project demonstrates a complete NLP pipeline for sentiment analysis on IMDb movie reviews using the TensorFlow Datasets library. The project is structured into separate modules for data preprocessing, model training, and model evaluation, ensuring a clean and maintainable codebase. By following this structure, you can easily extend and modify the project for other NLP tasks and datasets.
```
