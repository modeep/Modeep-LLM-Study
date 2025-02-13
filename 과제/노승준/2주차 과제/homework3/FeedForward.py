from torch import nn

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