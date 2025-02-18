import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Token Embedding
text = "Modeep is good"
text = text.lower()

vocab = {word: idx for idx, word in enumerate(text.split())}
print("Vocabulary:", vocab)
vocab_size = len(vocab)
embedding_dim = 128

token_ids = torch.tensor([vocab[word] for word in text.split()])

token_embedding = nn.Embedding(vocab_size, embedding_dim)
embedded = token_embedding(token_ids)

print("Original Text:", text.split())
print("Tokenized:", token_ids.tolist())
print("Embedded Vectors:\n", embedded)

# Layer Normalization
# 모델에 학습을 안정화하고 수렴속도를 높이는 정규화 기법
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# Scaled-dot Product Attention
# Self-Attention 연산이란 각각의 단어가 서로에게 어떤 연관성을 가지고 있는지를 파악하는 기술
class AttentionHead(nn.Module):
    def __init__(self, token_embed_dim, head_dim, mask=None):
        super().__init__()
        self.mask = mask
        self.weight_q = nn.Linear(token_embed_dim, head_dim)
        self.weight_k = nn.Linear(token_embed_dim, head_dim)
        self.weight_v = nn.Linear(token_embed_dim, head_dim)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output

    def forward(self, queries, keys, values):
        Q = self.weight_q(queries)
        K = self.weight_k(keys)
        V = self.weight_v(values)
        return self.scaled_dot_product_attention(Q, K, V, mask=self.mask)

# Multi-Head Attention
# CNN에서 여러 feature map을 사용하듯 self-attention을 여러 층으로 이루는 것
# 사람의 문장이기 때문에 복잡한게 많아 하나의 attetion으로는 성능이 떨어질 수 있어 여러개를 쓴다.
class MultiHeadAttention(nn.Module):
    def __init__(self, token_embed_dim, num_heads, mask=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = token_embed_dim // num_heads
        self.mask = mask

        self.heads = nn.ModuleList([
            AttentionHead(token_embed_dim, self.head_dim, mask)
            for _ in range(num_heads)
        ])
        self.output_linear = nn.Linear(num_heads * self.head_dim, token_embed_dim)

    def forward(self, query, key, value):
        head_outputs = [head(query, key, value) for head in self.heads]
        multi_head_output = torch.cat(head_outputs, dim=-1)
        return self.output_linear(multi_head_output)

# Feed Forward Layer
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn1 = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn1))
        attn2 = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + self.dropout(attn2))
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)

# 하이퍼파라미터
MODEL_DIM = 128
NUM_HEADS = 6
NUM_LAYERS = 2
D_FF = 64
DROPOUT = 0.1

# Model Initialization
decoder = Decoder(NUM_LAYERS, MODEL_DIM, NUM_HEADS, D_FF, DROPOUT)

# Input Preparation
src = embedded.unsqueeze(0)  # Add batch dimension
enc_output = torch.rand(1, src.size(1), MODEL_DIM)
src_mask = tgt_mask = None

output = decoder(src, enc_output, src_mask, tgt_mask)

print("Decoder Output:\n", output)
