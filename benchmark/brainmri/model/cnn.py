from torch import nn
import torch.nn.functional as F
from utils.fmodule import FModule


class Model(FModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, (4 ,4)),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 32, (4, 4)),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(32, 64, (4, 4)),
            nn.ReLU(True),
            nn.MaxPool2d((2,2)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
    
    def forward(self, x):
        # print("Here data shape", x.shape)
        rx = self.feature_extractor(x)
        x = self.classifier(rx)
        return x