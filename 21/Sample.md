Let's build a simple text classification program using pre-trained word embeddings and a neural network model. We'll use the Keras library for building and training the model. For this example, we'll use the GloVe embeddings and a sample dataset.

### Step-by-Step Guide

1. **Load Pre-trained GloVe Embeddings**
2. **Prepare the Dataset**
3. **Tokenize and Pad the Text Data**
4. **Create the Embedding Matrix**
5. **Build and Train the Neural Network Model**
6. **Evaluate the Model**

### Step 1: Load Pre-trained GloVe Embeddings

First, download the GloVe embeddings from [GloVe website](https://nlp.stanford.edu/projects/glove/) and place the file (e.g., `glove.6B.100d.txt`) in your working directory.

```python
import numpy as np

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
```

### Step 2: Prepare the Dataset

For simplicity, let's use a small sample dataset. In practice, you would use a larger dataset.

```python
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "Natural language processing is a fascinating field",
    "I enjoy learning about AI",
    "Machine learning can be applied to many domains"
]

labels = [1, 1, 1, 1, 1]  # For simplicity, let's assume all texts belong to the same category
```

### Step 3: Tokenize and Pad the Text Data

```python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index

# Pad sequences
max_sequence_length = 10
data = pad_sequences(sequences, maxlen=max_sequence_length)

print(f'Shape of data tensor: {data.shape}')
print(f'Found {len(word_index)} unique tokens.')
```

### Step 4: Create the Embedding Matrix

```python
embedding_dim = 100
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = glove_embeddings.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
```

### Step 5: Build and Train the Neural Network Model

```python
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1,
                    output_dim=embedding_dim,
                    weights=[embedding_matrix],
                    input_length=max_sequence_length,
                    trainable=False))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(data, np.array(labels), epochs=10, verbose=1)
```

### Step 6: Evaluate the Model

Since we used a very small dataset, the evaluation here is just for demonstration purposes. In practice, you would use a separate validation and test set.

```python
# Evaluate the model
loss, accuracy = model.evaluate(data, np.array(labels), verbose=0)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

### Conclusion

This simple program demonstrates how to use pre-trained GloVe word embeddings to classify text data into categories using a neural network model. In practice, you would use a larger dataset, more complex model architectures, and proper train-validation-test splits to build a robust text classification system.
