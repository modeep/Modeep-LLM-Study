import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)              
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  
        pe[:, 1::2] = torch.cos(position * div_term)  

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        
        return x + self.pe[:, :x.size(1)]



d_model = 16
seq_len = 10
batch_size = 2

x = torch.zeros(batch_size, seq_len, d_model)

pos_encoder = PositionalEncoding(d_model)
encoded_x = pos_encoder(x)

print("positional encoded:\n", encoded_x)

#sin , cos positional encoding 장점
#항상 positional encoding 값을 -1~1 사이의 값이 나온다.
#모델이 거리 기반의 패턴을 더 잘 인식할 수 있다.
#학습 데이터중 가장 긴 문장보다도 더 긴 문장이 실제 운영중에 들어와도 positional encoding이 에러없이 상대적인 인코딩값을 줄수있다는 점이다.
