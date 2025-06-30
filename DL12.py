# Program 12 Autoencoders
import torch
import torch.nn as nn
import torch.optim as optim
!pip install torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Data transformation
transform = transforms.Compose([
    transforms.ToTensor()
])
#  MNIST dataset
train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=False, transform=transform)
# Use subsets for quick training
train_subset = Subset(train_dataset, range(200))
test_subset = Subset(test_dataset, range(50))
# DataLoader
train_loader = DataLoader(train_subset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True)  # Added activation after last encoder layer
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, optimizer, and loss function
model = AutoEncoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
def train_model(num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for data in train_loader:
            img, _ = data
            img = img.view(img.size(0), -1).to(device)  # Flatten and move to device
            output = model(img)
            loss = criterion(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data in test_loader:
                img, _ = data
                img = img.view(img.size(0), -1).to(device)
                output = model(img)
                loss = criterion(output, img)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')


train_model(10)
