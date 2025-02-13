import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 위치 행렬 기본 세팅
        pe = torch.zeros((max_len, d_model))
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        den = torch.exp(-math.log(10000.0)*torch.arange(0, d_model, 2)/d_model)

        # 사인 & 코사인 함수를 위치 행렬에 적용
        pe[:, 0::2] = torch.sin(pos*den)
        pe[:, 1::2] = torch.cos(pos*den)

        # 더하기 연산을 위해 차원 추가
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
