from utils.fmodule import FModule
import torchvision

class Model(FModule):
    def __init__(self):
        super().__init__()
        return torchvision.models.efficientnet_b0(num_classes=200, pretrained=True)