"""
This is a non-official implementation of MOON proposed in 'Model-Contrastive
Federated Learning (https://arxiv.org/abs/2103.16257)'. The official
implementation is in https://github.com/QinbinLi/MOON. 
"""
from .fedbase import BasicServer, BasicClient
import copy
import torch
from utils import fmodule

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        if not "get_embedding" in dir(model):
            raise NotImplementedError("the model used by Moon should have the method `get_embedding` to obtain the intermediate result of forward")


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        # the temperature (tau) is set 0.5 as default
        self.tau = 0.5
        self.local_model = None
        self.contrastive_loss = ModelContrastiveLoss(self.tau)
        self.lossfunc = torch.nn.CrossEntropyLoss()
        self.mu = 0.1

    def train(self, model):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # init global model and local model
        global_model = copy.deepcopy(model).to(device)
        global_model.freeze_grad()
        
        if self.local_model:
            self.local_model.to(device)
        
        model = model.to(device)
        model.train()
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            
        for iter in range(self.epochs):
            for batch_id, batch_data in enumerate(data_loader):
                model.zero_grad()
                batch_data = self.calculator.data_to_device(batch_data, device)
                
                # calculate model contrastive loss
                z = model.get_embedding(batch_data[0])
                z_glob = global_model.get_embedding(batch_data[0])
                z_prev = self.local_model.get_embedding(batch_data[0]) if self.local_model else None
                
                loss_con = self.contrastive_loss(z, z_glob, z_prev)
                
                pred = model.decoder(z)
                loss_cls = self.lossfunc(pred, batch_data[1])
                
                loss = loss_cls + self.mu * loss_con
                loss.backward()
                optimizer.step()
                
        # update local model (move local model to CPU memory for saving GPU memory)
        self.local_model = copy.deepcopy(model).to(torch.device('cpu'))
        self.local_model.freeze_grad()
        return


class ModelContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ModelContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos = torch.nn.CosineSimilarity(dim=-1)
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, z, z_glob, z_prev):
        pos_sim = self.cos(z, z_glob)
        logits = pos_sim.reshape(-1, 1)
        if z_prev is not None:
            neg_sim = self.cos(z, z_prev)
            logits = torch.cat((logits, neg_sim.reshape(-1, 1)), dim=1)
        logits /= self.temperature
        return self.cross_entropy(logits, torch.zeros(z.size(0)).long().to(logits.device))