import math
import torch
from torch import nn
from torch.nn import functional as F

# scaled dot-product 연산 수행
def scaled_dot_product(query, key, value, mask=None):
    d_k = query.size(-1)
    # 연산식에 의해 연산 수행
    res = torch.matmul(query, torch.transpose(key, -2, -1)) / math.sqrt(d_k)
    if mask is not None:
        res.masked_fill(mask == 0, 1e-9)

    # 소프트맥스로 각 단어마다 확률 구하기
    res = F.softmax(res, dim=-1)
    # 실제 단어와 매칭
    return torch.matmul(res, value)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, embedding_dim, d_model):
        super().__init__()
        # 입력 벡터에 넣을 가중치 값
        self.w_q = nn.Linear(embedding_dim, d_model, bias=False)
        self.w_k = nn.Linear(embedding_dim, d_model, bias=False)
        self.w_v = nn.Linear(embedding_dim, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        # scaled dot-product 연산 수행
        return scaled_dot_product(self.w_q(query), self.w_k(key), self.w_v(value), mask)

# scaled dot-product 연산을 여러 모듈로 병렬 처리한 후 concate하고 선형 변환까지 하는 작업
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, h=8):
        super().__init__()
        self.layers = nn.ModuleList([ # h만큼 연산 레이어 증가
            ScaledDotProductAttention(embedding_dim, embedding_dim//h) for _ in range(h)
        ])

        self.fc_layer = nn.Linear(embedding_dim, embedding_dim) # 마지막 선형 변환 레이어

    # decoder에서는 query, key, value의 값이 같지 않아서 분리가 필요
    def forward(self, query, key, value, mask=None):
        out = torch.cat([layer(query, key, value, mask) for layer in self.layers], dim=-1) # 연산 병렬로 수행
        return self.fc_layer(out) # 마지막 선형 변환 후 반환

