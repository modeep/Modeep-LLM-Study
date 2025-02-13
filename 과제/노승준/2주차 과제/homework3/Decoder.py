import torch
from torch import nn
from torch.nn import functional as F
from homework3.MultiHeadAttention import MultiHeadAttention
from homework3.FeedForward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads=8, d_ff=2048):
        super().__init__()
        # decoder 아키텍처 구현 -> masked self attention -> cross attention -> feed forward
        self.masked_attention = MultiHeadAttention(embedding_dim, heads)
        self.cross_attention = MultiHeadAttention(embedding_dim, heads)
        self.feed_forward = FeedForward(embedding_dim, d_ff)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, x, encoder_input, mask, cross_mask=None):
        # masked self attention 수행
        x = x + self.masked_attention(x, x, x, mask)
        x = self.norm1(x) # 정규화 작업

        # cross attention 수행 -> encoder에서 생성한 벡터를 key, value로 입력을 넣는다.
        x = x + self.cross_attention(x, encoder_input, encoder_input, cross_mask)
        x = self.norm2(x) # 정규화 작업 수행

        # feed forward 수행 -> 늘렸다 줄이기
        x = x + self.feed_forward(x)
        out = self.norm3(x) # 정규화 작업 수행
        return out

class Decoder(nn.Module):
    def __init__(self, embedding_dim, heads=8, layers=6, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([ # layer 수만큼 쌓기
            DecoderLayer(embedding_dim, heads, d_ff) for _ in range(layers)
        ])
        self.fc_layer = nn.Linear(embedding_dim, embedding_dim) # 마지막 선형 변환

    def forward(self, x, encoder_input, mask, cross_mask=None):
        for layer in self.layers: # decoder를 layer 수만큼 수행
            x = layer(x, encoder_input, mask, cross_mask)
        x = self.fc_layer(x) # 마지막 선형 변환
        out = F.softmax(x, dim=-1) # 소프트맥스 함수 적용
        return out # 결과 출력
