import copy
from utils.fmodule import FModule
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
import numpy as np

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


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def flatten_model(model):
    """
    Flatten model (only linear layers)
    """
    ten = torch.cat([flatten_tensors(i) for i in model.parameters() if len(i.shape) == 2])
    return ten


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def unflatten_model(flat, model):
    count = 0
    l = []
    output = []
    for tensor in model.parameters():
        n = tensor.numel()
        output.append(flat[count: count + n].view_as(tensor))
        count += n
    output = tuple(output)
    temp = OrderedDict()
    for i, j in enumerate(model.state_dict().keys()):
        temp[j] = output[i]
    return temp


def sum_fim(fim1, fim2, p1, p2=None):
    """
    fims = [fim1, fim2]
    fim1 = [[A10, A11, A12], [G11, G12, G13]]
    fim2 = [[A20, A21, A22], [G21, G22, G23]]
    """
    if fim1 is None and fim2 is None:
        raise ValueError("Both can not be None")
    elif fim1 is None and fim2 is not None:
        return fim2
    elif fim1 is not None and fim2 is None:
        return fim1
    else:
        p2 = 1. - p1 if not p2 else p2
        A_res = [p1 * a1 + p2 * a2 for a1, a2 in zip(fim1[0], fim2[0])]
        G_res = [p1 * g1 + p2 * g2 for g1, g2 in zip(fim1[1], fim2[1])]
        return [A_res, G_res]


def sum_fims(fims, p=[]):
    """
    fims = [fim_i, ...]
    fim_i = [[Ai0, Ai1, Ai2], [Gi1, Gi2, Gi3]]
    """
    sump = np.sum(p)
    p = [pi/sump for pi in p]

    res = [[] for _ in fims[0]]
    for i in range(len(res)): # i=0,1
        l_res = res[i] # []
        for fim_i, pi in zip(fims, p):
            l_fim = fim_i[i] # [A0, A1, A2]
            if len(l_res):
                l_res = [o + pi * r for o, r in zip(l_res, l_fim)]
            else:
                l_res = copy.deepcopy([pi * item for item in l_fim])
        res[i] = l_res
    return res


def fim_to_device(fim, device):
    if isinstance(fim, torch.Tensor):
        return fim.to(device)
    else:
        fim = [fim_to_device(item, device) for item in fim]
        return fim