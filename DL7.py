import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Transform: resize to 224x224 and keep as 1-channel
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])

# Loaders
train = DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform), 64, True)
test = DataLoader(datasets.MNIST('data', train=False, download=True, transform=transform), 64)

# Load AlexNet and modify input/output
def get_alexnet_mnist():
    model = models.alexnet(pretrained=False)
    
    # Change first conv layer to accept 1 channel instead of 3
    model.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
    
    # Change classifier output to 10 classes
    model.classifier[6] = nn.Linear(4096, 10)
    
    return model

# Training and testing loop
def run():
    model = get_alexnet_mnist()
    opt = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(5):
        model.train()
        for x, y in train:
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()

    # Evaluate
    model.eval()
    c = t = 0
    for x, y in test:
        with torch.no_grad():
            preds = model(x).argmax(1)
            c += (preds == y).sum().item()
            t += y.size(0)
    return 100 * c / t

print("AlexNet on MNIST:", run())
