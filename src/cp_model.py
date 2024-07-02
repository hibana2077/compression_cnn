import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Compression(nn.Module):
    def __init__(self):
        super(Compression, self).__init__()
        self.compress = nn.Sequential(
            nn.AdaptiveAvgPool2d((5, 5)),
            nn.GELU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(9200, 10)
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)
        
    def forward(self, x):
        temp_stack = []
        x = F.gelu(self.conv1(x))
        temp_stack.append(self.compress(x))
        x = F.gelu(self.conv2(x))
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        x = self.conv3(x)
        temp_stack.append(self.compress(x))
        x = self.conv4(x)
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        total = torch.cat(temp_stack, dim=1)
        x = torch.cat([x, total], dim=1)
        x = F.layer_norm(x, x.size()[1:])
        x = self.flatten(x)
        x = F.gelu(x)
        x = self.fc(x)
        return x
    
if __name__ == '__main__':
    model = Compression()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(y.size())