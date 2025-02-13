from nltk.tokenize import word_tokenize
from torch import nn
import torch

def nltk_tokenization(sentence, vocab):
    token = []
    for s in sentence:
        words = word_tokenize(s.lower()) # 소문자롤 변경 후 토큰화
        token.append([vocab[word] for word in words]) # 마침표, 쉼표 유지
    return torch.Tensor(token).long() # 토큰화된 결과 반환 (batch_size, sentence_length)

# InputEmbedding 클래스
class InputEmbedding(nn.Module):
    def __init__(self, tokenizer, vocab, dim=512):
        super().__init__()
        self.embedding_layer = nn.Embedding(len(vocab), dim) # 임베딩 테이블 제작
        self.vocab = vocab # 단어 사전
        self.tokenizer = tokenizer # 입력 받은 토크나이저 사용

    def forward(self, x):
        x = self.tokenizer(x, self.vocab) # 토크나이저 적용
        return self.embedding_layer(x) # 임베딩 테이블 매핑
