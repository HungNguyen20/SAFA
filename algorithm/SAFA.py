"""
Implementation of SAFA: Handling Sparse and Scarce Data in Federated Learning with Accumulative Learning
"""

from .fedbase import BasicServer, BasicClient
from algorithm.utils.SAFA_supports.ewc import EWCv6
import os, torch, copy
import torch.nn.functional as F
import numpy as np
import random
from main import logger
import wandb

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


def get_penultimate_layer(model):
    modules = get_module_from_model(model)
    penul = modules[-1]._parameters['weight']
    return penul


def clustering(label_distribution: np.array):
    """
    return group_ids: np.array
    where group_ids[cid] is the group id of client cid
    """
    unq, count = np.unique(label_distribution, axis=0, return_counts=True)
    repeated_groups = unq[count > 0]

    group_ids = np.array([0 for i in range(label_distribution.shape[0])])
    group_id = 0
    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(label_distribution == repeated_group, axis=1))
        for cid in repeated_idx.ravel():
            group_ids[cid] = group_id
        group_id += 1
        
    return group_ids


class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.paras_name = ['lambda', 'offset', 'alpha']
        self.offset = option['offset']
        self.maxmimum_offset = option['offset']
        self.lambda_fct = option['lambda']
        self.alpha = option['alpha']
        self.EWCs = [EWCv6(id=i, model=copy.deepcopy(self.model), reg_coef=self.lambda_fct, lr=self.lr) for i in range(self.clients_per_round)]
        self.group_ids = None
        
        self.offset_clock = 0       # Do not change
        self.forget_tracking = torch.zeros([self.maxmimum_offset])
        self.gain_tracking = torch.zeros([self.maxmimum_offset])
        self.smoothness = 0.15
        
        self.p_clock = 0            # Do not change
        self.offset_retain = option['retain']
        self.advanced_stepsize = 5  # Tunable (but don't)
        
        self.confidence = 0         # Do not change
        self.confidence_thr = 10    # Tunable (but don't)
        return
    
    def preparation_round(self):
        grads = []
        for client in self.clients:
            local_grad = client.full_gradient(copy.deepcopy(self.model))
            temp = torch.sum(local_grad < 0, dim=1, keepdim=True).flatten()
            temp = (temp > 0) * 1.
            grads.append(temp)
        
        # Hard clustering
        label_distribution = torch.vstack(grads).cpu().numpy()
        self.group_ids = clustering(label_distribution)
        assert len(self.group_ids) == len(self.clients), "False indexing group ids!"
        for cid in range(len(self.group_ids)):
            self.clients[cid].group_id = self.group_ids[cid]
        return
    
    def run(self):
        self.preparation_round()
        super().run()
        return

    def iterate(self, t):
        self.selected_clients = sorted(self.sample())
        self.EWCs, gains, train_losses = self.communicate(self.selected_clients)
        print(logger.temp.format("Train loss:", np.mean(train_losses)))
        
        if not self.selected_clients: return
        
        self.update_forget_tracking()
        self.update_gain_tracking(gains)
        
        # Check best u
        if self.wandb and self.offset >= 3:
            cummulative_forget = torch.cumsum(self.forget_tracking[1:-1], dim=0)
            cummulative_gain = torch.cumsum(self.gain_tracking[1:-1], dim=0)
            knowledge_advantage = cummulative_gain - cummulative_forget
            max_val = knowledge_advantage.max()
            if max_val > 0:
                best_offset = knowledge_advantage.argmax() + 1 + 2
                if best_offset == self.offset:
                    best_offset += self.advanced_stepsize
            else:
                best_offset = 1
            wandb.log({"u best": best_offset}, t)
        
        impact_factors = [1.0 * ewc.trained_volume for ewc in self.EWCs]
        sum_q = np.sum(impact_factors)
        self.model = self.aggregate([ewc.model for ewc in self.EWCs], p = [q/sum_q for q in impact_factors])
        
        if (t > 0) and (t % self.offset == 0):
            self.p_clock += 1
            print("Updating the global model")
            importance = self.aggregate([ewc.importance for ewc in self.EWCs], p = [q/sum_q for q in impact_factors])
            for ewc in self.EWCs:
                ewc.model = (1 - self.alpha) * self.model + self.alpha * ewc.model
                ewc.importance = (1 - self.alpha) * importance + self.alpha * ewc.importance
                ewc.reset()
            
            if (self.p_clock % self.offset_retain == 0) and (self.offset >= 3):
                self.__adapt_offset()
                print(f"[Offset renew] At {t}: Offset renew to {self.offset}")
                self.offset_clock = 0
                self.p_clock = 0
        
        # Shuffle the continual learning models
        random.shuffle(self.EWCs)
        self.offset_clock = (self.offset_clock + 1) % self.offset  # offset clock = 0, 1, 2, ... (u-1)
        return

    def pack(self, client_id):
        cid = self.selected_clients.index(client_id)
        self.EWCs[cid].accumulate_data(self.client_vols[client_id])
        return {
            "model" : self.EWCs[cid],
        }
        
    def unpack(self, packages_received_from_clients):
        models = [cp["model"] for cp in packages_received_from_clients]
        gains = [cp["gain"] for cp in packages_received_from_clients]
        train_losses = [cp["train_loss"] for cp in packages_received_from_clients]
        return models, gains, train_losses
    
    def update_forget_tracking(self):
        forget_scale = torch.zeros_like(self.forget_tracking)
        forget_count = torch.zeros_like(self.forget_tracking)
        
        for ewc in self.EWCs:
            for traj in ewc.trajectories:
                u, forget = traj
                # print(u, "Len:", len(u), "Index:", len(u) - 1)
                index = len(u) -2
                forget_scale[index] += forget
                forget_count[index] += 1
        
        forget_count[forget_count == 0] = 1
        curr_forget = forget_scale / forget_count
        self.forget_tracking[forget_scale > 0] = (1 - self.smoothness) * self.forget_tracking[forget_scale > 0] + self.smoothness * curr_forget[forget_scale > 0]
        return
    
    def update_gain_tracking(self, gains):
        curr_gain = np.mean(gains)
        self.gain_tracking[self.offset_clock] = (1 - self.smoothness) * self.gain_tracking[self.offset_clock] + self.smoothness * curr_gain
        if self.offset_clock > self.offset:
            raise Exception("False here: offset_clock {} > offset {}".format(self.offset_clock, self.offset))
        return
        
    def __adapt_offset(self):
        
        cummulative_forget = torch.cumsum(self.forget_tracking[1:-1], dim=0)
        cummulative_gain = torch.cumsum(self.gain_tracking[1:-1], dim=0)
        knowledge_advantage = cummulative_gain - cummulative_forget
        
        print("Gain:", cummulative_gain)
        print("Forget:", cummulative_forget)
        print("Knowledge advantage:", knowledge_advantage)
        
        max_val = knowledge_advantage.max()
        best_offset = 3
        if max_val > 0:
            self.confidence = 0
            best_offset = knowledge_advantage.argmax() + 1 + 2
            if best_offset == self.offset:
                best_offset += self.advanced_stepsize
        else:
            self.confidence += 1
            
        if self.confidence < self.confidence_thr:
            self.offset = int(max(best_offset, int(self.offset/2)))
        else:
            self.offset = 1
        
        if self.offset > self.maxmimum_offset:
            self.maxmimum_offset = self.offset
            
            new_forget_tracking = torch.zeros([self.maxmimum_offset])
            new_forget_tracking[:self.forget_tracking.shape[0]] = self.forget_tracking
            self.forget_tracking = new_forget_tracking
            
            new_gain_tracking = torch.zeros([self.maxmimum_offset])
            new_gain_tracking[:self.gain_tracking.shape[0]] = self.gain_tracking
            self.gain_tracking = new_gain_tracking
            
        return
        
        
class Client(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super(Client, self).__init__(option, name, train_data, valid_data)
        self.train_c_epochs = self.epochs
        self.group_id = 0
        return
    
    def reply(self, svr_pkg):
        ewc = self.unpack(svr_pkg)
        
        test_acc_before_training, _ = self.test(ewc.model, device='cuda', dataflag='train')
        loss = self.train(ewc)       # Train the ewc
        test_acc_after_training, _ = self.test(ewc.model, device='cuda', dataflag='train')
        
        ewc.trackv2(self.group_id, test_acc_before_training, test_acc_after_training)
        cpkg = self.pack(ewc, test_acc_after_training - test_acc_before_training, loss)
        return cpkg
    
    def train(self, ewc : EWCv6):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        origin = copy.deepcopy(ewc).to(device)
        ewc.model.train()
        ewc.model = ewc.model.to(device)
        
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.optimizer_name, ewc.model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        losses = []
        
        for iter in range(self.epochs):
            batch_loss = []
            for batch_id, batch_data in enumerate(data_loader):
                ewc.model.zero_grad()
                clss_loss = self.calculator.get_loss(ewc.model, batch_data, device)
                reg_loss = ewc.compute_regularization(origin)
                loss = clss_loss + ewc.reg_coef * reg_loss
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.detach().cpu())
            losses.append(np.mean(batch_loss))
            
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=len(self.train_data))
        ewc.update_Fisher(data_loader, device)
        return np.mean(losses)
    
    def pack(self, model, gain, loss):
        return {
            "id" : int(self.name),
            "model" : model,
            "gain" : gain,
            "train_loss" : loss,
        }
        
    def full_gradient(self, model):
        """
        Return the full gradient of the classifier of the local dataset
        """
        model.train()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        data_loader = self.calculator.get_data_loader(self.train_data, batch_size=len(self.train_data))
        
        for batch_id, batch_data in enumerate(data_loader):
            model.zero_grad()
            loss = self.calculator.get_loss(model, batch_data, device)
            loss.backward()
        
        return get_penultimate_layer(model).grad