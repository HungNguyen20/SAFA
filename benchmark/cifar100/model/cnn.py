from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F

class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1600, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 100)
        
    def encoder(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1600)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
    
    def decoder(self, x):
        return self.fc3(x)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def pred_and_rep(self, x):
        e = self.encoder(x)
        o = self.decoder(e)
        return o, e
    
    def get_embedding(self, x):
        return self.encoder(x)
    
    
class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)