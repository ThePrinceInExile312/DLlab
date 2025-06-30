import torch, torch.nn as nn, torch.optim as optim

# Dummy input
X = torch.randn(16, 10, 8)              # (batch, sequence, features)
y = torch.randint(0, 2, (16, 1)).float()

# Build RNN model
rnn = nn.RNN(8, 32, batch_first=True)
fc = nn.Linear(32, 1)
params = list(rnn.parameters()) + list(fc.parameters())

# Loss and optimizer
loss_fn = nn.BCEWithLogitsLoss()
opt = optim.Adam(params)

# Training loop
for _ in range(10):
    out, _ = rnn(X)
    pred = fc(out[:, -1])
    loss = loss_fn(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"Loss: {loss.item():.4f}")
