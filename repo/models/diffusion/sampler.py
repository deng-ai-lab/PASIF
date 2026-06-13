import torch
import torch.nn as nn

class Sampler(nn.Module):
    def __init__(self, cfg, model, scl_dim=53, vec_dim=4):
        super(Sampler, self).__init__()

        self.node_feat_dim = cfg.get('node_feat_dim', 128)
        self.vec_feat_dim = cfg.get('vec_feat_dim', 128)
        self.edge_feat_dim = cfg.get('edge_feat_dim', 4)
        self.core_model = model

        self.scl_emb = nn.Linear(scl_dim, self.node_feat_dim)
        self.vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)

    def forward(self, x, vec, h, batch_idx):

        h = self.scl_emb(h)
        vec = self.vec_emb(vec.transpose(-1, -2)).transpose(-1, -2)

        scl_out, vec_out = self.core_model(x=x, vec=vec, h=h, batch_idx=batch_idx)

        return scl_out, vec_out
