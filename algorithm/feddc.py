"""
This is a non-official implementation of FedDC proposed in 'Federated Learning from Small Datasets' on (https://arxiv.org/abs/2110.03469).
"""

from .fedbase import BasicServer, BasicClient
import os, torch, copy
import torch.nn.functional as F
import numpy as np
import random


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['offset']
        self.offset = option['offset']
        self.cs = [copy.deepcopy(self.model) for _ in range(self.clients_per_round)]
        return
        
    def iterate(self, t):
        self.selected_clients = sorted(self.sample())
        self.cs, _ = self.communicate(self.selected_clients)
        random.shuffle(self.cs)
        
        if not self.selected_clients: return
        
        self.model = self.aggregate(self.cs, p = [1.0 for cid in self.selected_clients])
        if (t > 0) and (t % self.offset == 0):
            print("Updating the global model")
            self.cs = [copy.deepcopy(self.model) for c in self.cs]
            
        return

    def pack(self, client_id):
        return {
            "model" : self.cs[self.selected_clients.index(client_id)],
        }


class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        return
    
    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)      
        self.train(model)
        cpkg = self.pack(model, 0)
        return cpkg
