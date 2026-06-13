import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean


def log_1_min_a(a):
    return np.log(1 - np.exp(a) + 1e-40)

def to_torch_const(x):
    x = torch.from_numpy(x).float()
    x = nn.Parameter(x, requires_grad=False)
    return x

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def extract(coef, t, batch, ndim=2):
    out = coef[t][batch]
    if ndim == 1:
        return out
    elif ndim == 2:
        return out.unsqueeze(-1)
    elif ndim == 3:
        return out.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError('ndim > 3')

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    return sample_index

def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=-1)
    return kl

def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=-1)
