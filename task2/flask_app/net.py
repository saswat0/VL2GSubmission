import torch.nn as nn
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self,h1=96):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace = True)
        )

        self.drop1 = nn.Dropout(p=0.5)   

        self.fc1 = nn.Linear(32768, h1)
        self.drop2 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(h1, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.drop1(x)
        x = x.view(x.size(0),-1)

        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = self.fc2(x)

        return x