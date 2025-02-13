import math
import torch
from homework1.InputEmbedding import InputEmbedding, nltk_tokenization

word_vocab = {"after":0, "i":1, "finish":2, "my":3, "home":4,
    ",":5, "play":6, "favorite":7, "video":8,
    "games":9, "with":10, "friends":11, ".":12,
    "good":13, "homework":14
} # 임의로 만들어진 단어 사전
input_sentence = ["After I finish my homework, I play my favorite video games with my friends."]
embedding_dim = 4

# Input Embedding으로 만든 임베딩 벡터 값
input_embedding = InputEmbedding(nltk_tokenization, word_vocab, embedding_dim)
input_vector = input_embedding(input_sentence) # size: [batch_size, sentence_length, dim]


# 첫 번째 pe, 위치에 따라 pe 적용시키기 (선형)
def linear_pos(x, dim):
    return torch.arange(1, x.size(1)+1).unsqueeze(1).repeat(1, dim)

# 문장이 길어짐에 따라 점점 숫자가 커지면서 단어 간의 상관관계 정보를 파악하기 힘들다.
print(input_vector + linear_pos(input_vector, embedding_dim)) # 입력 벡터에 pe 더하기


# 처음은 0, 마지막은 1로 지정해서 1/단어 수 만큼 증가 시키기
def norm_pos(x, dim):
    return torch.arange(0, 1+1/x.size(1), 1/(x.size(1)-1)).unsqueeze(1).repeat(1, dim)

# 문장의 전체 길이를 알기 힘들고 문장 길이에 따라 같은 위치에 있는 값의 벡터가 바뀔 수 있다.
print(input_vector + norm_pos(input_vector, embedding_dim)) # 입력 벡터에 pe 더하기


# 1. 사인 & 코사인의 최댓값은 1, 최솟값은 -1로 단어 의미를 살릴 수 있다.
# 2. 문장이 길어져도 벡터 값의 차가 작아지지 않는다. → 주기함수의 특징
# 3. positional encoding의 특징으로 주기함수의 문제점을 여러 사인 & 코사인 함수를 사용해서 같은 위치는 그 벡터의 값으로,
# 서로 다른 위치는 서로 다른 벡터 값으로 표현할 수 있다.
def sin_cos_pos(d_model, max_len=5000):
    # 위치 행렬 벡터 생성
    pe = torch.zeros((max_len, d_model))
    # max_len만큼 함수를 적용시킬 벡터 생성
    pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    # pe 수식에 의한 계산 결과
    den = torch.exp(- math.log(10000.0) * torch.arange(0, d_model, 2) / d_model)

    # 위치 행렬에 적용
    pe[:, 0::2] = torch.sin(pos*den)
    pe[:, 1::2] = torch.cos(pos*den)
    pe = pe.unsqueeze(0)

    return pe

print(input_vector + sin_cos_pos(embedding_dim)[:, :input_vector.size(1)]) # 입력 벡터 길이만큼만 슬라이싱해서 결과값을 더함
