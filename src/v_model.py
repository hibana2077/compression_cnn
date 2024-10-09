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
        # 定義卷積層
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        
        # 定義池化層
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 定義全連接層
        self.fc1 = nn.Linear(256 * 8 * 8, 512)  # 假設輸入圖像大小為 28x28
        self.fc2 = nn.Linear(512, 10)  # 假設有 10 個類別
        
    def forward(self, x):
        # 應用第一個卷積層，然後是 ReLU 激活函數
        x = F.relu(self.conv1(x))
        
        # 應用第二個卷積層，然後是 ReLU 激活函數
        x = F.relu(self.conv2(x))
        
        # 應用第一個池化層
        x = self.pool(x)
        
        # 應用第三個卷積層，然後是 ReLU 激活函數
        x = F.relu(self.conv3(x))
        
        # 應用第四個卷積層，然後是 ReLU 激活函數
        x = F.relu(self.conv4(x))
        
        # 應用第二個池化層
        x = self.pool(x)
        
        # 將特徵圖展平
        x = x.view(-1, 256 * 8 * 8)
        
        # 應用全連接層
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
if __name__ == '__main__':
    net = VanillaCNN()
    x = torch.randn(4, 3, 32, 32)
    out = net(x)
    print(out.shape)