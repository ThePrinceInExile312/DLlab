import torch, torch.nn as nn, torch.optim as optim

text="Hello World this is simple text generation using LSTMs"
chars=sorted(set(text))
c2i={c:i for i,c in enumerate(chars)}
i2c={i:c for i,c in enumerate(chars)}

sq_ln=10
x=[[c2i[c] for c in text[i:i+sq_ln]] for i in range(len(text)-sq_ln)]
y=[c2i[text[i+sq_ln]] for i in range(len(text)-sq_ln)]

x_train=torch.tensor(x)
y_train=torch.tensor(y)

vocab=len(chars)
embed=nn.Embedding(vocab,8)
# lstm=nn.LSTM(8,32,batch_first=True)
lstm=nn.LSTM(8,32)

loss_fn=nn.CrossEntropyLoss()
fc=nn.Linear(32,vocab)
opt=optim.Adam(list(embed.parameters())+list(lstm.parameters())+list(fc.parameters()))
# opt=optim.Adam()

for _ in range(100):
    opt.zero_grad()
    out,_=lstm(embed(x_train))
    loss=loss_fn(fc(out[:,-1]),y_train)
    loss.backward()
    opt.step()

def gen(start):
    seq=[c2i[c] for c in start]
    for _ in range(50):
        x=torch.tensor([seq[-sq_ln:]])
        idx=fc(lstm(embed(x))[0][:,-1]).argmax().item()
        seq.append(idx)
    return start+''.join(i2c[c] for c in seq[sq_ln:])
print(gen("Hello Worl"))
