import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import PrunableNN, PrunableLinear
from utils import compute_sparsity, compute_sparsity_loss
import config


device = torch.device("cpu")


# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# SMALL SUBSET FOR FAST DEBUG
subset_indices = list(range(10000))
train_dataset = Subset(train_dataset, subset_indices)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)


def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return 100 * correct / total


for lambda_val in config.LAMBDA_VALUES:
    print(f"\nTraining with lambda = {lambda_val}")

    model = PrunableNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model(x)
            ce_loss = criterion(outputs, y)
            sparsity_loss = compute_sparsity_loss(model)

            loss = ce_loss + lambda_val * sparsity_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    acc = evaluate(model)
    sparsity = compute_sparsity(model)

    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            print("Gate mean:", gates.mean().item())

    print(f"Lambda: {lambda_val} | Accuracy: {acc:.2f}% | Sparsity: {sparsity:.2f}%")

import matplotlib.pyplot as plt

all_gates = []

for layer in model.modules():
    if isinstance(layer, PrunableLinear):
        gates = torch.sigmoid(layer.gate_scores)
        all_gates.extend(gates.detach().cpu().numpy().flatten())

plt.hist(all_gates, bins=50)
plt.title("Gate Value Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()
