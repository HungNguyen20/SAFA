import math
import torch
from utils.fmodule import FModule
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

# transform = transforms.Compose([            #[1]
#     transforms.Resize(256),                 #[2]
#     transforms.CenterCrop(224),                #[3]
#     transforms.ToTensor(),                     #[4]
#     transforms.Normalize(                      #[5]
#         mean=[0.485, 0.456, 0.406],                #[6]
#         std=[0.229, 0.224, 0.225]                  #[7]
#     )]
# )


class Model(FModule):
    def __init__(self):
        super(Model, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model._modules['fc'] = torch.nn.Linear(512, 10, bias=True)
        
    def forward(self, x):
        return self.model(x)
