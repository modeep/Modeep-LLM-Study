import torch
import torch.nn as nn

text = "Modeep is good"

text = text.lower()

vocab = {word: idx for idx, word in enumerate(text.split())}
print(vocab)
vocab_size = len(vocab)
embedding_dim = 8

token_ids = torch.tensor([vocab[word] for word in text.split()])  

token_embedding = nn.Embedding(vocab_size, embedding_dim)
embedded = token_embedding(token_ids)

print("원래 문자:", text.split())  
print("토큰화:", token_ids.tolist())
print("벡터:\n", embedded)
