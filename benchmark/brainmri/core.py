from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, CusTomTaskReader
import torchvision
from torch.utils.data import random_split
import os, torch

class TaskReader(CusTomTaskReader):
    def __init__(self, taskpath='', data_folder="../datasets/data/brain-tumor-mri-dataset"):
        transform= transforms.Compose([
            transforms.Resize(size= (150,150)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        train_dataset = datasets.ImageFolder(root=os.path.join(data_folder, "Training"), transform=transform, target_transform = None)
        test_dataset = datasets.ImageFolder(root=os.path.join(data_folder, "Testing"), transform=transform)
        super(TaskReader, self).__init__(taskpath, train_dataset, test_dataset)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
