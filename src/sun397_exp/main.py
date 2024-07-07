'''Train SUN397 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import timm
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import json
import argparse

from torch.utils.data import DataLoader,random_split
from models import *
from utils import progress_bar

info = {}
train_loss_history = []
train_acc_history = []
test_loss_history = []
test_acc_history = []
BATCH_SIZE = 128
parser = argparse.ArgumentParser(description='PyTorch SUN397 Training')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_full = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataset = torchvision.datasets.SUN397(
    root='../data', download=True, transform=transform_full)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
# net,net_name = Compression(), 'Compression'
net,net_name = CompressionNet_tiny(num_classes=397), 'CompressionNet_tiny'
# net,net_name = CompressionNet_small(num_classes=397), 'CompressionNet_small'
# net,net_name = CompressionNet_base(num_classes=397), 'CompressionNet_base'
# net,net_name = CompressionNet_medium(num_classes=397), 'CompressionNet_medium'
# net,net_name = CompressionNet_large(num_classes=397), 'CompressionNet_large'
# ResNet
# net, net_name = timm.create_model('resnet18', pretrained=False, num_classes=397), 'resnet18'
# net, net_name = timm.create_model('resnet34', pretrained=False, num_classes=397), 'resnet34'
# net, net_name = timm.create_model('resnet50', pretrained=False, num_classes=397), 'resnet50'
# net, net_name = timm.create_model('resnet101', pretrained=False, num_classes=397), 'resnet101'
# net, net_name = timm.create_model('resnet152', pretrained=False, num_classes=397), 'resnet152'
# EfficientNet
# net, net_name = timm.create_model('efficientnet_b0', pretrained=False, num_classes=397), 'efficientnet_b0'
# net, net_name = timm.create_model('efficientnet_b1', pretrained=False, num_classes=397), 'efficientnet_b1'
# net, net_name = timm.create_model('efficientnet_b2', pretrained=False, num_classes=397), 'efficientnet_b2'
# net, net_name = timm.create_model('efficientnet_b3', pretrained=False, num_classes=397), 'efficientnet_b3'
# net, net_name = timm.create_model('efficientnet_b4', pretrained=False, num_classes=397), 'efficientnet_b4'
# net, net_name = timm.create_model('efficientnet_b5', pretrained=False, num_classes=397), 'efficientnet_b5'
# net, net_name = timm.create_model('efficientnet_b6', pretrained=False, num_classes=397), 'efficientnet_b6'
# net, net_name = timm.create_model('efficientnet_b7', pretrained=False, num_classes=397), 'efficientnet_b7'
# Convnext
# net, net_name = timm.create_model('convnextv2_atto', pretrained=False, num_classes=397), 'convnextv2_atto'
# net, net_name = timm.create_model('convnextv2_femto', pretrained=False, num_classes=397), 'convnextv2_femto'
# net, net_name = timm.create_model('convnextv2_pico', pretrained=False, num_classes=397), 'convnextv2_pico'
# net, net_name = timm.create_model('convnextv2_nano', pretrained=False, num_classes=397), 'convnextv2_nano'
# net, net_name = timm.create_model('convnextv2_tiny', pretrained=False, num_classes=397), 'convnextv2_tiny'
# net, net_name = timm.create_model('convnextv2_small', pretrained=False, num_classes=397), 'convnextv2_small'
# net, net_name = timm.create_model('convnextv2_base', pretrained=False, num_classes=397), 'convnextv2_base'
# net, net_name = timm.create_model('convnextv2_large', pretrained=False, num_classes=397), 'convnextv2_large'
# net, net_name = timm.create_model('convnextv2_huge', pretrained=False, num_classes=397), 'convnextv2_huge'
# DenseNet
# net, net_name = timm.create_model('densenet121', pretrained=False, num_classes=397), 'densenet121'
# net, net_name = timm.create_model('densenet161', pretrained=False, num_classes=397), 'densenet161'
# net, net_name = timm.create_model('densenet169', pretrained=False, num_classes=397), 'densenet169'
# net, net_name = timm.create_model('densenet201', pretrained=False, num_classes=397), 'densenet201'
print('Number of parameters(M):', sum(p.numel() for p in net.parameters()) / 1e6)
print(net_name)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(net.parameters(), lr=4e-4, weight_decay=5e-4)
# optimizer = optim.AdamW(net.parameters(), lr=4e-4)
# optimizer = optim.AdamW(net.parameters(), lr=4e-4, weight_decay=0.05, betas=(0.9, 0.999))
optimizer = optim.SGD(net.parameters(), lr=0.1,momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    train_loss_history.append(train_loss / len(trainloader))
    train_acc_history.append(100.*correct/total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            
    test_loss_history.append(test_loss / len(testloader))
    test_acc_history.append(100.*correct/total)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+90):
    train(epoch)
    test(epoch)
    scheduler.step()

# Save the training information
info['epoch'] = epoch
info['batch_size'] = BATCH_SIZE
info['net'] = net_name
info['best_acc'] = best_acc
info['parameters'] = sum(p.numel() for p in net.parameters())
info['train_loss'] = train_loss_history
info['train_acc'] = train_acc_history
info['test_loss'] = test_loss_history
info['test_acc'] = test_acc_history
info['image_size'] = 32

# check the directory
if not os.path.isdir('info'):
    os.mkdir('info')
# save the information
with open(f'./info/{net_name}.json', 'w') as f:
    json.dump(info, f)