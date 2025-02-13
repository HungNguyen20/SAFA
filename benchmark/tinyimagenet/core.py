from torchvision import datasets, transforms
from benchmark.toolkits import ClassifyCalculator, CusTomTaskReader
import torchvision
from torch.utils.data import random_split
import os, torch

class TaskReader(CusTomTaskReader):
    def __init__(self, taskpath='', data_folder="./datasets/tiny-imagenet-200"):
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        alldata = torchvision.datasets.ImageFolder(os.path.join(data_folder, "train"), transform=transform)
        train_dataset, test_dataset = random_split(alldata, [80000, 20000], generator=torch.Generator().manual_seed(200200))
        super(TaskReader, self).__init__(taskpath, train_dataset, test_dataset)

class TaskCalculator(ClassifyCalculator):
    def __init__(self, device):
        super(TaskCalculator, self).__init__(device)
