### Week 4: Specialized Deep Learning Models and Techniques

**Day 22:** Sequence-to-Sequence Models

#### Sequence-to-Sequence Models: An Overview

Sequence-to-sequence (Seq2Seq) models are a type of neural network architecture designed to transform one sequence into another sequence. They are widely used in tasks such as machine translation, text summarization, and conversational agents.

#### Key Components of Seq2Seq Models

1. **Encoder**: Processes the input sequence and compresses the information into a context vector (also known as the thought vector).
2. **Decoder**: Takes the context vector and generates the output sequence.

#### Applications of Seq2Seq Models

- **Machine Translation**: Translating text from one language to another.
- **Text Summarization**: Summarizing long documents into shorter versions.
- **Chatbots**: Generating responses in conversational agents.
- **Speech Recognition**: Converting spoken language into text.

### Building a Simple Seq2Seq Model for a Translation Task

Let's build a simple Seq2Seq model for translating English sentences to French using Keras.

#### Step-by-Step Guide

1. **Prepare the Dataset**
2. **Preprocess the Data**
3. **Build the Seq2Seq Model**
4. **Train the Model**
5. **Evaluate the Model**

### Step 1: Prepare the Dataset

For simplicity, let's use a small dataset of English-French sentence pairs. In practice, you would use a larger dataset.

```python
# Sample dataset
data = [
    ("hello", "bonjour"),
    ("how are you", "comment ça va"),
    ("good morning", "bonjour"),
    ("good night", "bonne nuit"),
    ("thank you", "merci"),
    ("yes", "oui"),
    ("no", "non"),
    ("please", "s'il vous plaît"),
    ("sorry", "désolé"),
    ("goodbye", "au revoir")
]
```

### Step 2: Preprocess the Data

```python
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Separate the input and target texts
input_texts = [pair[0] for pair in data]
target_texts = ["\t" + pair[1] + "\n" for pair in data]  # Add start and end tokens

# Tokenize the input and target texts
input_tokenizer = Tokenizer()
input_tokenizer.fit_on_texts(input_texts)
input_sequences = input_tokenizer.texts_to_sequences(input_texts)
input_word_index = input_tokenizer.word_index

target_tokenizer = Tokenizer()
target_tokenizer.fit_on_texts(target_texts)
target_sequences = target_tokenizer.texts_to_sequences(target_texts)
target_word_index = target_tokenizer.word_index

# Pad the sequences
max_encoder_seq_length = max([len(seq) for seq in input_sequences])
max_decoder_seq_length = max([len(seq) for seq in target_sequences])

encoder_input_data = pad_sequences(input_sequences, maxlen=max_encoder_seq_length, padding='post')
decoder_input_data = pad_sequences(target_sequences, maxlen=max_decoder_seq_length, padding='post')

# Create decoder target data
decoder_target_data = np.zeros((len(data), max_decoder_seq_length, len(target_word_index) + 1), dtype='float32')
for i, seq in enumerate(target_sequences):
    for t, word_id in enumerate(seq):
        if t > 0:
            decoder_target_data[i, t - 1, word_id] = 1.0
```

### Step 3: Build the Seq2Seq Model

```python
from keras.models import Model
from keras.layers import Input, LSTM, Dense

# Define the encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(input_dim=len(input_word_index) + 1, output_dim=256)(encoder_inputs)
encoder_lstm = LSTM(256, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Define the decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(input_dim=len(target_word_index) + 1, output_dim=256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(len(target_word_index) + 1, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

### Step 4: Train the Model

```python
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=64,
          epochs=100,
          validation_split=0.2)
```

### Step 5: Evaluate the Model

To evaluate the model, we need to define the inference models for the encoder and decoder.

```python
# Define the encoder model for inference
encoder_model = Model(encoder_inputs, encoder_states)

# Define the decoder model for inference
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_embedding, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Function to decode a sequence
def decode_sequence(input_seq):
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_word_index['\t']

    # Sampling loop for a batch of sequences
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = target_tokenizer.index_word[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length or find stop character
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]

    return decoded_sentence

# Test the model
for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
```

### Conclusion

This example demonstrates how to build a simple Seq2Seq model for a translation task using Keras. The model consists of an encoder and a decoder, and it is trained on a small dataset of English-French sentence pairs. In practice, you would use a larger dataset and more complex model architectures to achieve better performance.
