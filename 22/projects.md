Let's create a comprehensive project that incorporates various concepts we've covered over the past 21 days. We'll build an end-to-end Natural Language Processing (NLP) pipeline that includes data preprocessing, feature extraction, model building, and evaluation. The project will focus on text classification, specifically sentiment analysis on a movie reviews dataset.

### Project: Sentiment Analysis on Movie Reviews

#### Steps:

1. **Data Collection and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Text Preprocessing**
4. **Feature Extraction using Word Embeddings**
5. **Model Building and Training**
6. **Model Evaluation**
7. **Model Inference**

### Step 1: Data Collection and Preprocessing

We'll use the IMDb movie reviews dataset, which is available through the `keras.datasets` module.

```python
import numpy as np
import pandas as pd
from keras.datasets import imdb

# Load the IMDb dataset
num_words = 10000  # Only consider the top 10,000 words
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Get the word index from the dataset
word_index = imdb.get_word_index()

# Reverse the word index to get the words from the indices
reverse_word_index = {value: key for key, value in word_index.items()}

# Decode the reviews back to text
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Decode a sample review
print(decode_review(x_train[0]))
```

### Step 2: Exploratory Data Analysis (EDA)

Let's perform some basic EDA to understand the dataset.

```python
import matplotlib.pyplot as plt

# Check the distribution of the labels
plt.hist(y_train, bins=2)
plt.title('Distribution of Sentiment Labels')
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.show()

# Check the length of the reviews
review_lengths = [len(review) for review in x_train]
plt.hist(review_lengths, bins=50)
plt.title('Distribution of Review Lengths')
plt.xlabel('Review Length')
plt.ylabel('Frequency')
plt.show()
```

### Step 3: Text Preprocessing

We'll pad the sequences to ensure they have the same length.

```python
from keras.preprocessing.sequence import pad_sequences

maxlen = 500  # Maximum length of the reviews

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(f'Shape of x_train: {x_train.shape}')
print(f'Shape of x_test: {x_test.shape}')
```

### Step 4: Feature Extraction using Word Embeddings

We'll use pre-trained GloVe embeddings for feature extraction.

```python
# Load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')
print(f'Loaded {len(glove_embeddings)} word vectors.')

# Create an embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if i < num_words:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
```

### Step 5: Model Building and Training

We'll build a simple LSTM model for sentiment analysis.

```python
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

model = Sequential()
model.add(Embedding(input_dim=num_words, output_dim=embedding_dim, weights=[embedding_matrix], input_length=maxlen, trainable=False))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
```

### Step 6: Model Evaluation

We'll evaluate the model on the test set.

```python
# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
```

### Step 7: Model Inference

We'll create a function to predict the sentiment of new reviews.

```python
def predict_sentiment(review):
    # Preprocess the review
    encoded_review = [word_index.get(word, 2) for word in review.split()]
    padded_review = pad_sequences([encoded_review], maxlen=maxlen)

    # Predict the sentiment
    prediction = model.predict(padded_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment

# Test the function
new_review = "This movie was fantastic! I really enjoyed it."
print(f'Review: {new_review}')
print(f'Sentiment: {predict_sentiment(new_review)}')
```

### Conclusion

This project demonstrates an end-to-end NLP pipeline for sentiment analysis on movie reviews. We covered data collection, preprocessing, feature extraction using word embeddings, model building, training, evaluation, and inference. This comprehensive project integrates various concepts learned over the past 21 days, providing a solid foundation for more advanced NLP tasks.
