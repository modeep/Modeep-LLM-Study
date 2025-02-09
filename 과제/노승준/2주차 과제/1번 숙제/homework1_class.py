from nltk.tokenize import word_tokenize
from torch import nn

def nltk_tokenization(sentence, vocab):
    words = word_tokenize(sentence.lower()) # 소문자로 변환 후 nltk 토크나이저 사용
    token = [vocab[word] for word in words] # 단어 사전으로 토큰화 진행
    return token # 결과 값 반환

# InputEmbedding 클래스
class InputEmbedding(nn.Module):
    def __init__(self, tokenizer, vocab, dim):
        super().__init__()
        embedding_layer = nn.Embedding(len(vocab), dim) # 임베딩 테이블 제작
        self.vocab = vocab # 단어 사전
        self.embedding_weight = embedding_layer.weight # 임베딩 테이블 가중치 저장
        self.tokenizer = tokenizer # 입력 받은 토크나이저 사용

    def forward(self, x):
        x = self.tokenizer(x, self.vocab) # 토크나이저 적용
        return self.embedding_weight[x] # 임베딩 테이블 매핑

word_vocab = {"after":0, "i":1, "finish":2, "my":3, "home":4,
    ",":5, "play":6, "favorite":7, "video":8,
    "games":9, "with":10, "friends":11, ".":12,
    "good":13, "homework":14
} # 임의로 만들어진 단어 사전
input_sentence = "After I finish my homework, I play my favorite video games with my friends."

input_embedding = InputEmbedding(nltk_tokenization, word_vocab, 512) # 토크나이저, 단어 사전, 벡터 차원 입력
print(input_embedding(input_sentence)) # 원하는 문장 입력 후 결과 출력