from nltk.tokenize import word_tokenize
from torch import nn, Tensor

input_sentence = ["After I finish my homework, I play my favorite video games with my friends."]
vocab = {"after":0, "i":1, "finish":2, "my":3, "home":4,
    ",":5, "play":6, "favorite":7, "video":8,
    "games":9, "with":10, "friends":11, ".":12,
    "good":13, "homework":14
} # 임의로 만들어진 단어 사전

# 토큰화 1 -> 단순하게 띄어쓰기 단위로 나누기
def simple_tokenization(sentence):
    token = []
    for s in sentence:
        words = s.lower().split(" ") # 소문자로 변경 후 단어 나누기
        words = [word if word[-1] not in [",", ".", "?", "!", "~"] else word[:-1] for word in words] # 마침표, 쉼표 제거
        token.append([vocab[word] for word in words])
    return Tensor(token).long() # 토큰화된 결과 반환 (batch_size, sentence_length)

# 토큰화 2 -> NLTK의 word_tokenize 사용
def nltk_tokenization(sentence):
    token = []
    for s in sentence:
        words = word_tokenize(s.lower()) # 소문자롤 변경 후 토큰화
        token.append([vocab[word] for word in words]) # 마침표, 쉼표 유지
    return Tensor(token).long() # 토큰화된 결과 반환 (batch_size, sentence_length)

# 토큰화된 결과 값 출력
simple = simple_tokenization(input_sentence)
nltk_token = nltk_tokenization(input_sentence)
print(simple)
print(nltk_token)

# 임베딩 테이블을 만드는 과정 -> pytorch nn.embedding 사용
# 단어 사전의 길이와 변환하고 싶은 차원 수(보통 512 사용)를 파라미터로 입력
embedding_layer = nn.Embedding(len(vocab), 3)

embedded = embedding_layer(nltk_token) # 임베딩 적용
print(embedded) # 최종 임베딩 결과 출력