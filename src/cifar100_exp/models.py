import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Compression_T(nn.Module): #tiny
    def __init__(self):
        super(Compression_T, self).__init__()
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.final_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.se1 = SELayer(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.se2 = SELayer(32)
        self.conv3 = nn.Conv2d(32, 96, 3)
        self.se3 = SELayer(96)
        self.conv4 = nn.Conv2d(96, 384, 3)
        self.se4 = SELayer(384)
        self.conv5 = nn.Conv2d(912, 96, 1)
        self.se5 = SELayer(96)
        
    def forward(self, x):
        temp_stack = []
        x = self.conv1(x)
        x = self.se1(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.conv2(x)
        x = self.se2(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.pool(x)
        temp_stack.append(self.compress(x))
        x = self.conv4(x)
        x = self.se4(x)
        temp_stack.append(self.compress(x))
        if x.size()[-2:] == (2, 2):
            x = self.pool(x)
        if x.size()[-2:] != (2, 2):
            x = self.final_pool(x)
        total = torch.cat(temp_stack, dim=1)
        x = torch.cat([x, total], dim=1)
        x = self.conv5(x)
        x = self.se5(x)
        x = F.layer_norm(x, x.size()[1:])
        x = self.flatten(x)
        x = F.gelu(x)
        return x # (4, 384)
    
class Compression_S(nn.Module): #small
    def __init__(self):
        super(Compression_S, self).__init__()
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.GELU()
        )
        self.final_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.se1 = SELayer(32)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.se2 = SELayer(64)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.se3 = SELayer(96)
        self.conv4 = nn.Conv2d(96, 384, 3)
        self.se4 = SELayer(384)
        self.conv5 = nn.Conv2d(960, 128, 1)
        self.se5 = SELayer(128)
        
    def forward(self, x):
        temp_stack = []
        x = self.conv1(x)
        x = self.se1(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.conv2(x)
        x = self.se2(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.pool(x)
        temp_stack.append(self.compress(x))
        x = self.conv4(x)
        x = self.se4(x)
        temp_stack.append(self.compress(x))
        if x.size()[-2:] != (3, 3):
            x = self.final_pool(x)
        total = torch.cat(temp_stack, dim=1)
        x = torch.cat([x, total], dim=1)
        x = self.conv5(x)
        x = self.se5(x)
        x = F.layer_norm(x, x.size()[1:])
        x = self.flatten(x)
        x = F.gelu(x)
        return x # (4, 1152)
    
class Compression_M(nn.Module): #medium
    def __init__(self):
        super(Compression_M, self).__init__()
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.GELU()
        )
        self.final_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.se1 = SELayer(64)
        self.conv2 = nn.Conv2d(64, 96, 3)
        self.se2 = SELayer(96)
        self.conv3 = nn.Conv2d(96, 384, 3)
        self.se3 = SELayer(384)
        self.conv4 = nn.Conv2d(384, 768, 3)
        self.se4 = SELayer(768)
        self.conv5 = nn.Conv2d(2080, 256, 1)
        self.se5 = SELayer(256)
        
    def forward(self, x):
        temp_stack = []
        x = self.conv1(x)
        x = self.se1(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.conv2(x)
        x = self.se2(x)
        x = F.relu(x)
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.se3(x)
        x = self.pool(x)
        temp_stack.append(self.compress(x))
        x = self.conv4(x)
        x = self.se4(x)
        temp_stack.append(self.compress(x))
        if x.size()[-2:] != (3, 3):
            x = self.final_pool(x)
        total = torch.cat(temp_stack, dim=1)
        x = torch.cat([x, total], dim=1)
        x = self.conv5(x)
        x = self.se5(x)
        x = F.layer_norm(x, x.size()[1:])
        x = self.flatten(x)
        x = F.gelu(x)
        return x # (4, 2304)

class CompressionNet_tiny(nn.Module):
    def __init__(self):
        super(CompressionNet_tiny, self).__init__()
        self.compression = Compression_T()
        self.cls = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(64, 100)
        )

    def forward(self, x):
        x = self.compression(x)
        x = self.cls(x)
        return x
    
class CompressionNet_small(nn.Module):
    def __init__(self):
        super(CompressionNet_small, self).__init__()
        self.compression = Compression_S()
        self.cls = nn.Sequential(
            nn.Linear(1152, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 100)
        )

    def forward(self, x):
        x = self.compression(x)
        x = self.cls(x)
        return x

class CompressionNet_medium(nn.Module):
    def __init__(self):
        super(CompressionNet_medium, self).__init__()
        self.compression = Compression_M()
        self.cls = nn.Sequential(
            nn.Linear(2304, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 100)
        )

    def forward(self, x):
        x = self.compression(x)
        x = self.cls(x)
        return x
    
if __name__ == '__main__':
    model = CompressionNet_medium()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(y.size())