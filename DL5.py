import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

train = DataLoader(datasets.MNIST('data',train=True,download=True,transform=transforms.ToTensor()),64,True)
test = DataLoader(datasets.MNIST('data',train=False,download=True,transform=transforms.ToTensor()),64)

def model(bn=True,drop=True):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784,256),
        nn.BatchNorm1d(256) if bn else nn.Identity(),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256,10)
    )

def run1(bn=False,drop=False):
    mod=model(bn,drop)
    opt=optim.Adam(mod.parameters())
    loss=nn.CrossEntropyLoss()
    accl = []
    epochs = 5
    for epoch in range(epochs):
        mod.train()
        for x,y in train:
            opt.zero_grad()
            loss(mod(x),y).backward()
            opt.step()
        
        # Fixed: Evaluate after each epoch, not just once
        c=t=0
        mod.eval()
        with torch.no_grad():
            for x,y in test:
                c += (mod(x).argmax(1) == y).sum().item()
                t += y.size(0)
        acc = 100 * c / t
        print(f"adam epoch {epoch+1} acc {acc}")
        accl.append(acc)
    return accl

def run2(bn=False,drop=False):
    mod=model(bn,drop)
    opt=optim.SGD(mod.parameters())
    loss=nn.CrossEntropyLoss()
    accl = []
    epochs = 5
    for epoch in range(epochs):
        mod.train()
        for x,y in train:
            opt.zero_grad()
            loss(mod(x),y).backward()
            opt.step()
        
        # Fixed: Evaluate after each epoch, not just once
        c=t=0
        mod.eval()
        with torch.no_grad():
            for x,y in test:
                c += (mod(x).argmax(1) == y).sum().item()
                t += y.size(0)
        acc = 100 * c / t
        print(f"SGD epoch {epoch+1} acc {acc}")
        accl.append(acc)
    return accl

adaml = run1()
sgdl = run2()
plt.plot(adaml,label="adam")
plt.plot(sgdl,label="SGD")
plt.legend()
plt.show()

# print("With BN    :", run(bn=True))
# print("With Drop  :", run(drop=True))
# print("BN + Drop  :", run1(bn=True, drop=True))
