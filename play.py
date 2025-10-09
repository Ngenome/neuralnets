import tiktoken
import torch


enc = tiktoken.get_encoding("gpt2")
with open("input.txt", "r")as f:
    text = f.read()
text = text[:1000]
tokens = enc.encode(text)
print(tokens[:24])
B, T = 4,32
buf = torch.tensor(tokens[:B*T+1])
x = buf[:-1].view(B,T)
y = buf[1:].view(B, T)

model=GPT(GPTConfig())
model.to(device)
logits = model(x)

print(logits.shape)
import sys; sys.exit(0)