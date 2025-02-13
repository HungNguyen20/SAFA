from torch import nn
from utils.fmodule import FModule
import torch
import torch.nn.functional as F


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc1 = nn.Linear(1600, 384, bias=True)
        self.fc2 = nn.Linear(384, 192, bias=True)
        self.fc3 = nn.Linear(192, 10, bias=True)
        
    def decoder(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x.flatten(1))
    
    def pred_and_rep(self, x):
        e = self.encoder(x)
        o = self.decoder(e.flatten(1))
        return o, e.flatten(1)
    
    def get_embedding(self, x):
        return self.encoder(x).flatten(1)
    

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)