import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader

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
def run(bn=False,drop=False):
    mod=model(bn,drop)
    opt=optim.Adam(mod.parameters())
    loss=nn.CrossEntropyLoss()
    for _ in range(5):
        mod.train()
        for x,y in train:
            opt.zero_grad()
            loss(mod(x),y).backward()
            opt.step()
    c=t=0
    mod.eval()
    for x,y in test:
        c+=(mod(x).argmax(1)==y).sum().item()
        t+=y.size(0)
    return 100*c/t
print("None       :", run())
print("With BN    :", run(bn=True))
print("With Drop  :", run(drop=True))
print("BN + Drop  :", run(bn=True, drop=True))
