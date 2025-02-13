from homework1.InputEmbedding import InputEmbedding, nltk_tokenization
from homework2.PositionalEncoding import PositionalEncoding
from homework3.Encoder import Encoder

word_vocab = {"after":0, "i":1, "finish":2, "my":3, "home":4,
    ",":5, "play":6, "favorite":7, "video":8,
    "games":9, "with":10, "friends":11, ".":12,
    "good":13, "homework":14
} # 임의로 만들어진 단어 사전
input_sentence = ["After I finish my homework, I play my favorite video games with my friends."]
embedding_dim = 512

input_embedding = InputEmbedding(nltk_tokenization, word_vocab, embedding_dim)
input_vector = input_embedding(input_sentence) # 토큰화, 토큰 임베딩 적용
print(input_vector)

pe = PositionalEncoding(embedding_dim) # 위치 벡터 적용
print(pe(input_vector))

encoder = Encoder(embedding_dim) # encoder 적용
print(encoder(input_vector))