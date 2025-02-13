import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import random
from utils.fmodule import FModule


def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res


class EWC(nn.Module):
    def __init__(self, id, model: FModule, reg_coef, loss_fn=torch.nn.CrossEntropyLoss(), lr=0.001):
        super().__init__()
        self.reg_coef = reg_coef
        self.loss_fn = loss_fn
        self.model = model
        self.id = id
        
        self.importance = model.zeros_like()
        self.trained_volume = 0
        return
    
    def accumulate_data(self, size):
        self.trained_volume += size
        return
    
    def forward(self, x):
        return self.model(x)
    
    def compute_regularization(self, origin):
        """
        origin: EWC
        """
        reg_loss = 0
        for p, q, k in zip(self.model.parameters(), origin.model.parameters(), self.importance.parameters()):
            if p.grad is not None:
                reg_loss += (k * (p - q)**2).sum()
        return reg_loss

    def update_Fisher(self, dataloader, device):
        # Update the diag fisher information
        mode = self.training
        self.eval()
        
        self_module = get_module_from_model(self.model)
        importance_module = get_module_from_model(self.importance)

        # Accumulate the square of gradients
        for i, (input, target) in enumerate(dataloader):
            input = input.to(device)
            target = target.to(device)
            preds = self.forward(input)
            loss = self.loss_fn(preds, target)
            self.model.zero_grad()
            loss.backward()
            
            for i in range(len(self_module)):
                if self_module[i].weight.grad is not None:
                    importance_module[i].weight.data += (self_module[i].weight.grad.data.to(device))**2 * len(input) / len(dataloader)
                    importance_module[i].bias.data += (self_module[i].bias.grad.data.to(device))**2 * len(input) / len(dataloader)
            
        self.train(mode=mode)
        return
    
    
class EWCv5(EWC):
    def __init__(self, id, model, reg_coef, loss_fn=torch.nn.CrossEntropyLoss(), lr=0.001):
        super().__init__(id, model, reg_coef, loss_fn, lr)
        self.reset()
        return
    
    def trackv2(self, cluster_id, arrival_acc, departure_acc):
        """
        trajectories = [(trajectory, forget), ...]
        """
        if (cluster_id in self.visited_cluster_list):
            tmp = self.visited_cluster_list.copy()
            tmp.reverse()
            start_index = len(tmp) - tmp.index(cluster_id) - 1
            
            trajectory = self.visited_cluster_list[start_index:].copy()
            trajectory.append(cluster_id)
            
            forget = self.recorded_departure_acc[start_index] - arrival_acc
            self.trajectories.append((trajectory, forget))
        
        self.visited_cluster_list.append(cluster_id)
        self.recorded_arrival_acc.append(arrival_acc)
        self.recorded_departure_acc.append(departure_acc)
        return
    
    def trackv1(self, cluster_id, arrival_acc, departure_acc):
        """
        trajectories = [(u, forget), ...]
        """
        if (cluster_id in self.visited_cluster_list):
            tmp = self.visited_cluster_list.copy()
            tmp.reverse()
            start_index = len(tmp) - tmp.index(cluster_id) - 1
            
            trajectory = self.visited_cluster_list[start_index:].copy()
            trajectory.append(cluster_id)
            
            forget = self.recorded_departure_acc[start_index] - arrival_acc
            self.trajectories.append((len(trajectory), forget))
        
        self.visited_cluster_list.append(cluster_id)
        self.recorded_arrival_acc.append(arrival_acc)
        self.recorded_departure_acc.append(departure_acc)
        return
        
    def reset(self):
        self.importance = self.importance.zeros_like()
        self.trained_volume = 0
        self.visited_cluster_list = []
        self.recorded_arrival_acc = []         # Record the inference acc, before training
        self.recorded_departure_acc = []       # Record the inference acc, after training
        self.trajectories = []
        return


class EWCv6(EWCv5):
    def __init__(self, id, model, reg_coef, loss_fn=torch.nn.CrossEntropyLoss(), lr=0.001):
        super().__init__(id, model, reg_coef, loss_fn, lr)
        
    def reset(self):
        self.trained_volume = 0
        self.visited_cluster_list = []
        self.recorded_arrival_acc = []         # Record the inference acc, before training
        self.recorded_departure_acc = []       # Record the inference acc, after training
        self.trajectories = []
        return