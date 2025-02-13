from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, CusTomTaskReader
import torchvision
from torch.utils.data import random_split
import os, torch

class TaskReader(CusTomTaskReader):
    def __init__(self, taskpath='', data_folder="../datasets/data/medical-mnist"):
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        dataset=datasets.ImageFolder(root=data_folder,transform=transform)
        train_dataset, test_dataset = random_split(dataset, [40000, len(dataset) - 40000], generator=torch.Generator().manual_seed(200200))
        super(TaskReader, self).__init__(taskpath, train_dataset, test_dataset)


class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
