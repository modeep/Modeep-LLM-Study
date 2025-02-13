import math
import torch
from torch import nn
from torch.nn import functional as F
from homework1.InputEmbedding import InputEmbedding, nltk_tokenization
from homework2.PositionalEncoding import PositionalEncoding

word_vocab = {"after":0, "i":1, "finish":2, "my":3, "home":4,
    ",":5, "play":6, "favorite":7, "video":8,
    "games":9, "with":10, "friends":11, ".":12,
    "good":13, "homework":14
} # 임의로 만들어진 단어 사전
input_sentence = ["After I finish my homework, I play my favorite video games with my friends."]
embedding_dim = 512

input_embedding = InputEmbedding(nltk_tokenization, word_vocab, embedding_dim)
input_vector = input_embedding(input_sentence) # 토큰화, 토큰 임베딩 적용

pe = PositionalEncoding(embedding_dim) # 위치 벡터 적용
input_vector = pe(input_vector)

# scaled dot-product 연산 수행
def scaled_dot_product(query, key, value, mask=None):
    d_k = query.size(-1)
    # 연산식에 의해 연산 수행
    res = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)
    if mask:
        res.masked_fill(mask == 0, 1e-9)

    # 소프트맥스로 각 단어마다 확률 구하기
    res = F.softmax(res, dim=-1)
    # 실제 단어와 매칭
    return torch.matmul(res, value)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, d_model, mask=None):
        super().__init__()
        # 입력 벡터에 넣을 가중치 값
        self.w_q = nn.Linear(embedding_dim, d_model, bias=False)
        self.w_k = nn.Linear(embedding_dim, d_model, bias=False)
        self.w_v = nn.Linear(embedding_dim, d_model, bias=False)
        self.mask = mask

    def forward(self, x):
        # scaled dot-product 연산 수행
        return scaled_dot_product(self.w_q(x), self.w_k(x), self.w_v(x), self.mask)

# scaled dot-product 연산을 여러 모듈로 병렬 처리한 후 concate하고 선형 변환까지 하는 작업
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, h=8, mask=None):
        super().__init__()
        self.layers = nn.ModuleList([ # h만큼 연산 레이어 증가
            ScaledDotProductAttention(embedding_dim, embedding_dim//h, mask) for _ in range(h)
        ])

        self.fc_layer = nn.Linear(embedding_dim, embedding_dim) # 마지막 선형 변환 레이어

    def forward(self, x):
        out = torch.cat([layer(x) for layer in self.layers], dim=-1) # 연산 병렬로 수행
        return self.fc_layer(out) # 마지막 선형 변환 후 반환

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, d_ff=2048):
        super().__init__()
        # linear로 차원을 늘렸다가 줄이는 역할 수행 -> 정확도 향상
        self.fc_layer1 = nn.Linear(embedding_dim, d_ff)
        self.fc_layer2 = nn.Linear(d_ff, embedding_dim)

    def forward(self, x):
        out = self.fc_layer1(x) # 차원 늘리기
        out = self.fc_layer2(out) # 차원 줄이기
        return out

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads=8, d_ff=2048, mask=None):
        super().__init__()
        # encoder 하나에 들어가는 연산: multi-head attention -> add&norm -> feed forward -> add&norm
        self.multi_head_attention = MultiHeadAttention(embedding_dim, heads, mask)
        self.feed_forward = FeedForward(embedding_dim, d_ff)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # 처음 self-attention 수행
        x = x + self.multi_head_attention(x) # multi-head attention 수행 전 벡터와 합연산
        x = self.norm1(x) # 정규화

        # feed forward 수행
        x = x + self.feed_forward(x) # feed forward 수행 전 벡터와 합연산
        out = self.norm2(x) # 정규화
        return out

class Encoder(nn.Module):
    def __init__(self, embedding_dim, heads=8, layers=6, d_ff=2048, mask=None):
        super().__init__()
        self.layers = nn.ModuleList([ # encoder를 heads만큼 쌓기
            EncoderLayer(embedding_dim, heads, d_ff, mask) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim) # 마지막 정규화

    def forward(self, x):
        for layer in self.layers: # 각 encoder에서 순차적으로 연산
            x = layer(x)
        return self.norm(x) # 마지막 정규화 수행
