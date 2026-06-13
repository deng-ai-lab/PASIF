import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from repo.utils.protein.constants import *
from repo.modules.common import compose_context
from repo.modules.e3nn.gvptransformer import GVPTransformer
from torch.distributions import Categorical


class Actor(nn.Module):
    def __init__(self, cfg, vec_dim=1):
        super(Actor, self).__init__()

        # self.max_mask_num = cfg.get('max_mask_num', 10)
        self.max_gen_num = cfg.get('max_gen_num', 15)

        self.num_classes = cfg.get('num_atomtype', 14)
        self.node_feat_dim = cfg.encoder.get('node_feat_dim', 128)
        self.vec_feat_dim = cfg.encoder.get('vec_feat_dim', 32)
        self.edge_feat_dim = cfg.encoder.get('edge_feat_dim', 4)
        if cfg.encoder.get('type', 'gvptransformer') == 'gvptransformer':
            self.encoder = GVPTransformer(cfg.encoder)
        else:
            raise ValueError('Other encoder isn\'t supproted!')

        if cfg.embedder.residue.get('type', 'linear') == 'linear':
            self.protein_atom_emb = nn.Linear(len(atomic_numbers) + 1, 
                                              cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types protein embedder isn\'t supproted!')
        
        if cfg.embedder.atom.get('type', 'linear') == 'linear':
            self.ligand_atom_emb = nn.Linear(self.num_classes, 
                                             cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types ligand embedder isn\'t supproted!')

        self.vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)
        
        self.mask_head = nn.Sequential(nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, 2), nn.Softmax(dim=-1), )
        self.gen_head = nn.Sequential(nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, self.max_gen_num+1), nn.Softmax(dim=-1), )
        
    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):

        h_lig = self.ligand_atom_emb(h_lig)
        h_rec = self.protein_atom_emb(h_rec)

        lig_flag = torch.ones_like(batch_idx_lig).float()
        rec_flag = torch.zeros_like(batch_idx_rec).float()
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec, 'h': h_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)

        x = context_composed['x']
        h = torch.concat((context_composed['h'], context_composed['lig_flag'].unsqueeze(-1)), dim=-1)
        vec = self.vec_emb(x.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)

        scl_emb, vec_emb = self.encoder(x=x, vec=vec, h=h, batch_idx=batch_idx)
        scl_emb = scl_emb[context_composed['lig_flag'].bool()]

        scl_aggr = scatter_mean(scl_emb, batch_idx_lig, dim=0)   # (B, node_feat_dim)
        mask_dist = self.mask_head(scl_emb)  # (Nlig, 2)
        gen_dist = self.gen_head(scl_aggr)   # (B, max_gen_num)

        return mask_dist, gen_dist

class Critic(nn.Module):
    def __init__(self, cfg, vec_dim=1):
        super(Critic, self).__init__()

        self.num_classes = cfg.get('num_atomtype', 14)
        self.node_feat_dim = cfg.encoder.get('node_feat_dim', 128)
        self.vec_feat_dim = cfg.encoder.get('vec_feat_dim', 32)
        self.edge_feat_dim = cfg.encoder.get('edge_feat_dim', 4)
        if cfg.encoder.get('type', 'gvptransformer') == 'gvptransformer':
            self.encoder = GVPTransformer(cfg.encoder)
        else:
            raise ValueError('Other encoder isn\'t supproted!')

        if cfg.embedder.residue.get('type', 'linear') == 'linear':
            self.protein_atom_emb = nn.Linear(len(atomic_numbers) + 1, 
                                              cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types protein embedder isn\'t supproted!')
        
        if cfg.embedder.atom.get('type', 'linear') == 'linear':
            self.ligand_atom_emb = nn.Linear(self.num_classes, 
                                             cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types ligand embedder isn\'t supproted!')

        self.vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)
        
        self.critic_head = nn.Sequential(nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, self.node_feat_dim), nn.ReLU(), 
                                       nn.Linear(self.node_feat_dim, 1), )
        
    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):

        h_lig = self.ligand_atom_emb(h_lig)
        h_rec = self.protein_atom_emb(h_rec)

        lig_flag = torch.ones_like(batch_idx_lig).float()
        rec_flag = torch.zeros_like(batch_idx_rec).float()
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec, 'h': h_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)

        x = context_composed['x']
        h = torch.concat((context_composed['h'], context_composed['lig_flag'].unsqueeze(-1)), dim=-1)
        vec = self.vec_emb(x.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)

        scl_emb, vec_emb = self.encoder(x=x, vec=vec, h=h, batch_idx=batch_idx)

        scl_aggr = scatter_mean(scl_emb, batch_idx, dim=0)   # (B, node_feat_dim)
        critic = self.critic_head(scl_aggr)   # (B, 1)

        return critic, batch_idx

class ConstantActor(nn.Module):
    def __init__(self, atom_num, mask_idx=None, temp=1.0):
        super(ConstantActor, self).__init__()

        self.temp = temp
        if mask_idx is None:
            self.hid_param = nn.Parameter(torch.randn((atom_num, 2)))
        else:
            self.constant_init(mask_idx=mask_idx, atom_num=atom_num)

    def constant_init(self, atom_num, mask_idx):
        
        init_param = torch.zeros((atom_num, 2))
        init_param[:, 0] = 1
        init_param[mask_idx, 0] = 0
        init_param[mask_idx, 1] = 1
        self.hid_param = nn.Parameter(init_param)
    
    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):
        
        B = batch_idx_lig.max() + 1
        mask_prob = F.softmax(self.hid_param/self.temp, dim=-1)

        return mask_prob.repeat(B, 1)
    
class ConstantCritic(nn.Module):
    def __init__(self, ):
        super(ConstantCritic, self).__init__()

        self.hid_param = nn.Parameter(torch.randn((1)))
    
    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):
        
        B = batch_idx_lig.max() + 1

        return self.hid_param.repeat(B, 1)