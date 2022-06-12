import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride = stride, padding = padding),
            nn.ReLU()
        )

    def forward(self, x):
        return self.model(x)

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.model(x)
  
class DropletNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(1, 8, 3),
            nn.MaxPool2d(3, 2),
            ConvBlock(8, 8, 3),
            nn.MaxPool2d(3, 2),
            ConvBlock(8, 8, 3),
            nn.MaxPool2d(3, 2),
            ConvBlock(8, 8, 3),
            nn.MaxPool2d(5, 3),
            nn.Flatten(),
            Linear(2888, 128),
            Linear(128, 16),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.model(x)