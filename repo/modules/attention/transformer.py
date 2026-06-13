import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim):
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.linear_q = nn.Linear(input_dim, hid_dim, bias=False)
        self.linear_k = nn.Linear(input_dim, hid_dim, bias=False)
        self.linear_v = nn.Linear(input_dim, out_dim, bias=False)
        self.norm_fact = 1 / math.sqrt(hid_dim)

    def forward(self, x, batch_idx):
        
        batch_equal = batch_idx.unsqueeze(1) == batch_idx.unsqueeze(0)
        attn_mask = torch.zeros_like(batch_equal)
        attn_mask[~batch_equal] = float('-inf')
        q = self.linear_q(x)
        k = self.linear_q(x)
        v = self.linear_v(x)

        dist = torch.mm(q, k.transpose(-1, -2)) * self.norm_fact
        dist = dist + attn_mask
        dist = F.softmax(dist, dim=-1)

        att = torch.mm(dist, v)

        return att

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, head_num):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        att_list = []
        for i in range(head_num):
            att_list.append(SelfAttention(input_dim, hid_dim, out_dim))
        self.mul_att = nn.ModuleList(att_list)

        self.out_linear = nn.Linear(head_num*out_dim, out_dim)

    def forward(self, x, batch_idx, res=True):

        feat_list = []
        for att in self.mul_att:
            feat_list.append(att(x, batch_idx))
        z = torch.concat(feat_list, dim=-1)
        if res:
            out = x + self.out_linear(z)
        else:
            out = self.out_linear(z)
        return out