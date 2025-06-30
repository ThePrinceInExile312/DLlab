import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MNIST loaders (subset for speed)
transform = transforms.ToTensor()
train_data = Subset(datasets.MNIST("./data", train=True, transform=transform, download=True), range(200))
test_data = Subset(datasets.MNIST("./data", train=False, transform=transform, download=True), range(50))
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10)

# Simple Autoencoder using nn.Sequential
autoencoder = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.Linear(256, 28*28),
    nn.Sigmoid()
).to(device)

optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    autoencoder.train()
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(device).view(imgs.size(0), -1)
        output = autoencoder(imgs)
        loss = criterion(output, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluation
    autoencoder.eval()
    test_loss = 0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(device).view(imgs.size(0), -1)
            output = autoencoder(imgs)
            loss = criterion(output, imgs)
            test_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}")
