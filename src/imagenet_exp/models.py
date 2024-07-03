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
        self.name = 'CompressionNet'
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 128, 3)# maybe change dim 96 -> 384
        
    def forward(self, x):
        temp_stack = []
        x = F.relu(self.conv1(x))
        temp_stack.append(self.compress(x))
        x = F.relu(self.conv2(x))
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        x = self.conv3(x)
        temp_stack.append(self.compress(x))
        x = self.conv4(x)
        temp_stack.append(self.compress(x))
        x = self.pool(x)
        total = torch.cat(temp_stack, dim=1)# maybe can add more conv layers
        x = torch.cat([x, total], dim=1)
        x = F.layer_norm(x, x.size()[1:])
        x = self.flatten(x)
        x = F.gelu(x)
        return x # (4,9200)
    
class DueCompression(nn.Module):
    def __init__(self):
        super(DueCompression, self).__init__()
        self.name = 'Due_CompressionNet'
        self.compressionA = Compression()
        self.compressionB = Compression()
        self.fusion1 = nn.Linear(9200*2, 1024)
        self.fusion2 = nn.Linear(1024, 100)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x_A = self.compressionA(x)
        x_B = self.compressionB(x)
        x = torch.cat([x_A, x_B], dim=1)
        x = self.fusion1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fusion2(x)
        return x

class QuadCompression(nn.Module):
    def __init__(self):
        super(QuadCompression, self).__init__()
        self.name = 'Quad_CompressionNet'
        self.compressionA = Compression()
        self.compressionB = Compression()
        self.compressionC = Compression()
        self.compressionD = Compression()
        self.fusion1 = nn.Linear(9200*4, 1024)
        self.fusion2 = nn.Linear(1024, 100)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x_A = self.compressionA(x)
        x_B = self.compressionB(x)
        x_C = self.compressionC(x)
        x_D = self.compressionD(x)
        x = torch.cat([x_A, x_B, x_C, x_D], dim=1)
        x = self.fusion1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fusion2(x)
        return x
    
class OctoCompression(nn.Module):
    def __init__(self):
        super(OctoCompression, self).__init__()
        self.name = 'Octo_CompressionNet'
        self.compressionA = Compression()
        self.compressionB = Compression()
        self.compressionC = Compression()
        self.compressionD = Compression()
        self.compressionE = Compression()
        self.compressionF = Compression()
        self.compressionG = Compression()
        self.compressionH = Compression()
        self.fusion1 = nn.Linear(9200*8, 1024)
        self.fusion2 = nn.Linear(1024, 100)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x_A = self.compressionA(x)
        x_B = self.compressionB(x)
        x_C = self.compressionC(x)
        x_D = self.compressionD(x)
        x_E = self.compressionE(x)
        x_F = self.compressionF(x)
        x_G = self.compressionG(x)
        x_H = self.compressionH(x)
        x = torch.cat([x_A, x_B, x_C, x_D, x_E, x_F, x_G, x_H], dim=1)
        x = self.fusion1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fusion2(x)
        return x
    
class HexaCompression(nn.Module):
    def __init__(self):
        super(HexaCompression, self).__init__()
        self.name = 'Hexa_CompressionNet'
        self.compressionA = Compression() #1
        self.compressionB = Compression() #2
        self.compressionC = Compression() #3
        self.compressionD = Compression() #4
        self.compressionE = Compression() #5
        self.compressionF = Compression() #6
        self.compressionG = Compression() #7
        self.compressionH = Compression() #8
        self.compressionI = Compression() #9
        self.compressionJ = Compression() #10
        self.compressionK = Compression() #11
        self.compressionL = Compression() #12
        self.compressionM = Compression() #13
        self.compressionN = Compression() #14
        self.compressionO = Compression() #15
        self.compressionP = Compression() #16
        self.fusion1 = nn.Linear(9200*16, 1024)
        self.fusion2 = nn.Linear(1024, 100)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x_A = self.compressionA(x)
        x_B = self.compressionB(x)
        x_C = self.compressionC(x)
        x_D = self.compressionD(x)
        x_E = self.compressionE(x)
        x_F = self.compressionF(x)
        x_G = self.compressionG(x)
        x_H = self.compressionH(x)
        x_I = self.compressionI(x)
        x_J = self.compressionJ(x)
        x_K = self.compressionK(x)
        x_L = self.compressionL(x)
        x_M = self.compressionM(x)
        x_N = self.compressionN(x)
        x_O = self.compressionO(x)
        x_P = self.compressionP(x)
        x = torch.cat([x_A, x_B, x_C, x_D, x_E, x_F, x_G, x_H, x_I, x_J, x_K, x_L, x_M, x_N, x_O, x_P], dim=1)
        x = self.fusion1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fusion2(x)
        return x

if __name__ == '__main__':
    model = HexaCompression()
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(f"Parameter count(M): {sum(p.numel() for p in model.parameters()) / 1e6}")
    print(y.size())