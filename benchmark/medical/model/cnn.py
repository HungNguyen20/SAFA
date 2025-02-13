from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule

class Model(FModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(3136, 64)
        self.lin2 = nn.Linear(64, 6)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x