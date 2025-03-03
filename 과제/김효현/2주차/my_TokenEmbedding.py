from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
import numpy as np

text = 'hello world'
text.lower()

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
vocab = tokenizer.word_index
print('단어사전 : ', vocab)

token_ids = tokenizer.texts_to_sequences([text])[0]
print("토큰화:", token_ids)

vocab_size = len(vocab) + 1
embedding_dim = 8

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=len(token_ids))
])

model.compile(optimizer='adam', loss='mse')

# 4. 임베딩 결과 확인
embedded = model.predict(np.array([token_ids]))

print("\n원래 문자:", text.split())  
print("토큰화:", token_ids)
print("벡터:\n", embedded[0])