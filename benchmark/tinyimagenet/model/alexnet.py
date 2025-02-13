from utils.fmodule import FModule
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F

class Model(FModule):
    def __init__(self):
        super().__init__()
        model = torchvision.models.alexnet(num_classes=1000, pretrained=True)
        # print(model)
        self.feature_extractor = model.features
        self.avg_pool = model.avgpool
        
        # freeze
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        
        for p in self.avg_pool.parameters():
            p.requires_grad = False
        
        # learnable classifier
        self.fc1 = nn.Linear(9216, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 200)
        return
    
    def encoder(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        return torch.flatten(x, 1)
    
    def decoder(self, rx):
        x = F.relu(self.fc1(rx))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def forward(self, x):
        rx = self.encoder(x)
        return self.decoder(rx)
        
