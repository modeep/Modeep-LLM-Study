from torch import nn
from homework3.FeedForward import FeedForward
from homework3.MultiHeadAttention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, heads=8, d_ff=2048):
        super().__init__()
        # encoder 하나에 들어가는 연산: multi-head attention -> add&norm -> feed forward -> add&norm
        self.multi_head_attention = MultiHeadAttention(embedding_dim, heads)
        self.feed_forward = FeedForward(embedding_dim, d_ff)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        # 처음 self-attention 수행
        x = x + self.multi_head_attention(x, x, x, mask) # multi-head attention 수행 전 벡터와 합연산
        x = self.norm1(x) # 정규화

        # feed forward 수행
        x = x + self.feed_forward(x) # feed forward 수행 전 벡터와 합연산
        out = self.norm2(x) # 정규화
        return out

class Encoder(nn.Module):
    def __init__(self, embedding_dim, heads=8, layers=6, d_ff=2048):
        super().__init__()
        self.layers = nn.ModuleList([ # encoder를 heads만큼 쌓기
            EncoderLayer(embedding_dim, heads, d_ff) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim) # 마지막 정규화

    def forward(self, x, mask=None):
        for layer in self.layers: # 각 encoder에서 순차적으로 연산
            x = layer(x, mask)
        return self.norm(x) # 마지막 정규화 수행
