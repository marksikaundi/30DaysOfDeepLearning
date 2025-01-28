Certainly! Let's dive into a more complex example of the attention mechanism, specifically focusing on the Transformer model, which has revolutionized the field of NLP. The Transformer model uses a self-attention mechanism extensively and is the foundation for models like BERT, GPT, and many others.

### Transformer Model with Self-Attention

The Transformer model, introduced by Vaswani et al. in the paper "Attention is All You Need," uses self-attention mechanisms to process input sequences in parallel, making it highly efficient and effective for various NLP tasks.

#### Key Components of the Transformer Model

1. **Self-Attention Mechanism**: Allows each position in the input sequence to attend to all other positions, enabling the model to capture dependencies regardless of their distance in the sequence.
2. **Multi-Head Attention**: Enhances the model's ability to focus on different parts of the input sequence by using multiple attention heads.
3. **Positional Encoding**: Adds information about the position of each token in the sequence since the model processes the entire sequence in parallel.
4. **Feed-Forward Neural Networks**: Applied to each position in the sequence independently and identically.
5. **Layer Normalization and Residual Connections**: Help stabilize and improve the training process.

#### Implementing the Transformer Model

Let's implement a simplified version of the Transformer model using PyTorch.

##### Step 1: Define the Scaled Dot-Product Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: [batch_size, num_heads, seq_len, d_k]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        return output, attn_weights
```

##### Step 2: Define the Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.fc(attn_output)

        return output, attn_weights
```

##### Step 3: Define the Positional Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
```

##### Step 4: Define the Transformer Encoder Layer

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, src_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        # Feed-forward
        ff_output = self.fc2(F.relu(self.fc1(src)))
        src = src + self.dropout(ff_output)
        src = self.norm2(src)

        return src
```

##### Step 5: Define the Transformer Encoder

```python
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        src = self.positional_encoding(src)
        src = self.dropout(src)

        for layer in self.layers:
            src = layer(src, src_mask)

        return src
```

##### Step 6: Define the Transformer Decoder Layer

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(trg, trg, trg, trg_mask)
        trg = trg + self.dropout(attn_output)
        trg = self.norm1(trg)

        # Cross-attention
        attn_output, _ = self.cross_attn(trg, memory, memory, memory_mask)
        trg = trg + self.dropout(attn_output)
        trg = self.norm2(trg)

        # Feed-forward
        ff_output = self.fc2(F.relu(self.fc1(trg)))
        trg = trg + self.dropout(ff_output)
        trg = self.norm3(trg)

        return trg
```

##### Step 7: Define the Transformer Decoder

```python
class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, d_model, num_heads, num_layers, d_ff, dropout=0.1, max_len=5000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, memory, trg_mask=None, memory_mask=None):
        trg = self.embedding(trg) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, dtype=torch.float32))
        trg = self.positional_encoding(trg)
        trg = self.dropout(trg)

        for layer in self.layers:
            trg = layer(trg, memory, trg_mask, memory_mask)

        output = self.fc_out(trg)

        return output
```

##### Step 8: Define the Complete Transformer Model

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads, num_encoder_layers, d_ff, dropout, max_len)
        self.decoder = TransformerDecoder(trg_vocab_size, d_model, num_heads, num_decoder_layers, d_ff, dropout, max_len)

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(trg, memory, trg_mask, src_mask)
        return output
```

This implementation provides a more complex and comprehensive example of the attention mechanism within the Transformer model. The Transformer model leverages self-attention and multi-head attention to process sequences in parallel, making it highly effective for various NLP tasks.

### Conclusion

The Transformer model, with its self-attention mechanism, has significantly advanced the field of NLP. By understanding and implementing the Transformer model, you can build powerful models for tasks such as machine translation, text summarization, and more. This example demonstrates the complexity and effectiveness of the attention mechanism in deep learning.
