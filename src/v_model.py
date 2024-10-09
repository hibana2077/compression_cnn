import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        

        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.conv2(x))
        
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        
        x = F.relu(self.conv4(x))
        
        x = self.pool(x)
        
        x = x.view(-1, 256 * 8 * 8)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
if __name__ == '__main__':
    net = VanillaCNN()
    x = torch.randn(4, 3, 32, 32)
    out = net(x)
    print(out.shape)
