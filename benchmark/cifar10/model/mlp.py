from torch import nn
from utils.fmodule import FModule
import torch.nn.functional as F
import torch

def get_kronecker_product_diag(A, B):
    dA = torch.diag(A).unsqueeze(0)
    dB = torch.diag(B).unsqueeze(0)
    return torch.flatten(dA.T * dB).tolist()
        

class Model(FModule):
    def __init__(self, dim_in=3*32*32, dim_hidden=256, dim_out=10):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.fc2 = nn.Linear(dim_hidden, dim_hidden)
        self.fc3 = nn.Linear(dim_hidden, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def decoder(self, x):
        x = self.fc3(x)
        return x

    def encoder(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def get_embedding(self, x):
        return self.encoder(x)
    
    # def forward_fim(self, x):
    #     self.a0 = x.view(x.shape[0], -1)
        
    #     self.s1 = self.fc1(self.a0)
    #     self.a1 = F.relu(self.s1)
        
    #     self.s2 = self.fc2(self.a1)
    #     self.a2 = F.relu(self.s2)
        
    #     self.s3 = self.fc3(self.a2)
        
    #     self.FIM_params = [self.s1, self.s2, self.s3]
    #     self.FIM_as = [self.a0, self.a1, self.a2]
        
    #     return self.s3
    
    # def compute_fisher(self, X:torch.Tensor, Y:torch.Tensor, loss_fn=torch.nn.KLDivLoss(reduction='batchmean'), device='cuda'):    
    #     self = self.to(device)
    #     X, Y = X.to(device), Y.to(device)
        
    #     pred = self.forward_fim(X)
    #     loss = loss_fn(F.log_softmax(pred, dim=1), F.softmax(F.one_hot(Y, 10) * 1.0, dim=1))    
        
    #     grads = torch.autograd.grad(loss, self.FIM_params, create_graph=False, retain_graph=False)
    #     g1, g2, g3 = grads
    #     a0, a1, a2 = self.FIM_as
        
    #     A0 = torch.mean(a0.unsqueeze(2) @ a0.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
    #     A1 = torch.mean(a1.unsqueeze(2) @ a1.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
    #     A2 = torch.mean(a2.unsqueeze(2) @ a2.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
    #     G1 = torch.mean(g1.unsqueeze(2) @ g1.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
    #     G2 = torch.mean(g2.unsqueeze(2) @ g2.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
    #     G3 = torch.mean(g3.unsqueeze(2) @ g3.unsqueeze(2).transpose(1,2), dim=0).detach().cpu()
        
    #     self = self.cpu()
    #     return [A0, A1, A2], [G1, G2, G3]
    
    
    # def compute_fisher_diag(self, X:torch.Tensor, Y:torch.Tensor, loss_fn=torch.nn.KLDivLoss(reduction='batchmean'), device='cuda'): 
   
    #     A, G = self.compute_fisher(X, Y, loss_fn=loss_fn, device=device)
    #     A0, A1, A2 = A
    #     G1, G2, G3 = G
        
    #     diagonal = get_kronecker_product_diag(A0, G1) + get_kronecker_product_diag(A1, G2) + get_kronecker_product_diag(A2, G3)
    #     return torch.Tensor(diagonal)
        

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.cross_entropy(output, target)