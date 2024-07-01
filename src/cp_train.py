import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import timm
import timm.optim as tiopt

from torch.utils.data import DataLoader

from cp_model import Compression

# 設置設備（GPU如果可用，否則使用CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 數據預處理和加載
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=4, shuffle=False)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# net = timm.create_model('resnet18').to(device)
net = Compression().to(device)

# print model parameters number
num_params = sum(p.numel() for p in net.parameters())
print(f'Number of parameters(M): {num_params / 1e6:.2f}')

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = tiopt.Lookahead(optim.AdamW(net.parameters(), lr=0.0001))

# 訓練模型
confidence = {}
correctness = {}
print("Start training")
epochs = 10
for epoch in range(epochs):
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{epochs}], loss: {running_loss / len(trainloader):.3f}')

print('Finished Training')

# 在測試集上評估模型
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')