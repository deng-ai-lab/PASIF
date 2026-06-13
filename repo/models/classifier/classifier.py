import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import coalesce
from repo.utils.protein.constants import *
from repo.modules.common import compose_context
from repo.modules.e3nn.gvptransformer import GVPTransformer
from torch.distributions import Categorical

from repo.modules.attention.transformer import MultiHeadAttention
from repo.modules.e3nn.gvptransformer import AttentionInteractionBlockVN
from repo.modules.common import GaussianSmearing, VecExpansion
from repo.modules.gvp.gvn import GVLinear, VNLeakyReLU, MessageModule, GVPerceptronVN


class PropPredictor(nn.Module):
    def __init__(self, cfg, vec_dim=1, model_type='reg', protein_dim=None):
        super(PropPredictor, self).__init__()

        if protein_dim is None:
            protein_dim = len(atomic_numbers) + 1
        self.model_type = model_type
        self.num_classes = cfg.get('num_atomtype', 14)
        self.node_feat_dim = cfg.encoder.get('node_feat_dim', 128)
        self.vec_feat_dim = cfg.encoder.get('vec_feat_dim', 32)
        self.edge_feat_dim = cfg.encoder.get('edge_feat_dim', 4)
        if model_type == 'cls':
            cfg.encoder.num_classes = 2
        if cfg.encoder.get('type', 'gvptransformer') == 'gvptransformer':
            self.encoder = GVPTransformer(cfg.encoder)
        else:
            raise ValueError('Other encoder isn\'t supproted!')

        if cfg.embedder.residue.get('type', 'linear') == 'linear':
            self.protein_atom_emb = nn.Linear(protein_dim, 
                                              cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types protein embedder isn\'t supproted!')
        
        if cfg.embedder.atom.get('type', 'linear') == 'linear':
            self.ligand_atom_emb = nn.Linear(self.num_classes, 
                                             cfg.embedder.get('emb_dim', 128)-1)
        else:
            raise ValueError('Other types ligand embedder isn\'t supproted!')
        
        self.vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)

    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, just_lig=False):

        if just_lig:
            h_lig = self.ligand_atom_emb(h_lig)
            lig_flag = torch.ones_like(batch_idx_lig).float()
            x = x_lig
            h = torch.concat((h_lig, lig_flag.unsqueeze(-1)), dim=-1)
            vec = self.vec_emb(x.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)
            batch_idx = batch_idx_lig
            scl_out, vec_out = self.encoder(x=x, vec=vec, h=h, batch_idx=batch_idx)
        else:
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

            scl_out, vec_out = self.encoder(x=x, vec=vec, h=h, batch_idx=batch_idx)

        return scl_out, vec_out, batch_idx

    def predict(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, just_lig=False):

        if h_lig.ndim == 1:
            h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()
        atom_scl_out, _, batch_idx = self(x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, just_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)
        if self.model_type == 'cls':
            final_exp_pred = F.softmax(final_exp_pred, dim=-1)[:, 1]
        else:
            final_exp_pred = final_exp_pred[:, 0]
        
        return final_exp_pred
    
    def get_loss(self, prop, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, just_lig=False):

        h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()
        atom_scl_out, _, batch_idx = self(x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, just_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)
        if self.model_type == 'cls':
            final_exp_pred = F.softmax(final_exp_pred, dim=-1)
            prop_onehot = F.one_hot(prop.long(), num_classes=2).float()
            loss = - (prop_onehot * torch.log(final_exp_pred+1.e-5)).sum(-1).mean()
        else:
            final_exp_pred = final_exp_pred[:, 0]
            loss = torch.abs(final_exp_pred - prop).mean()

        return {'loss': loss}, {'pred': final_exp_pred, 'label': prop}
    
    def get_noise_loss(self, prop, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig, discrete='uniform', just_lig=False):
        
        B = batch_idx_lig.max() + 1
        h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()

        noise_scale = torch.rand(size=[B], device=prop.device)
        true_prob = torch.rand(size=[B], device=prop.device)
        true_prob[true_prob>=0.5] = 1
        true_prob[true_prob<0.5] = 0
        noise_scale[true_prob.bool()] = 0.
        noise_scale_lig = noise_scale[batch_idx_lig][:, None]

        x_lig_noise = (1 - noise_scale_lig) * x_lig + noise_scale_lig * torch.randn_like(x_lig)
        if discrete == 'uniform':
            h_lig_noise = (1 - noise_scale_lig) * h_lig + noise_scale_lig * torch.ones_like(h_lig) / self.num_classes
            # noise_dist = Categorical(probs=h_lig_noise)
            # h_lig_noise = noise_dist.sample()
            # h_lig_noise = F.one_hot(h_lig_noise, num_classes=self.num_classes).float()
        elif discrete == 'gauss':
            h_lig_noise = (1 - noise_scale_lig) * h_lig + noise_scale_lig * torch.randn_like(h_lig)
        elif discrete == 'absorbing':
            absorbing_state = torch.zeros_like(h_lig)
            absorbing_state[:, 0] = 1.
            h_lig_noise = (1 - noise_scale_lig) * h_lig + noise_scale_lig * absorbing_state
            # noise_dist = Categorical(probs=h_lig_noise)
            # h_lig_noise = noise_dist.sample()
            # h_lig_noise = F.one_hot(h_lig_noise, num_classes=self.num_classes).float()

        noise_prop = (1 - noise_scale) * prop

        atom_scl_out, _, batch_idx = self(x_rec, x_lig_noise, h_rec, h_lig_noise, batch_idx_rec, batch_idx_lig, just_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)[:, 0]
        loss = torch.abs(final_exp_pred - noise_prop).mean()

        return {'loss': loss}, {'pred': final_exp_pred, 'label': noise_prop}

    def eval_noise_loss(self, prop, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):
        
        B = 32
        h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()

        noise_scale = torch.linspace(0, 1, B).to(prop.device)
        batch_idx_lig = torch.concat([(batch_idx_lig+1)*i for i in range(B)], dim=0)
        batch_idx_rec = torch.concat([(batch_idx_rec+1)*i for i in range(B)], dim=0)
        x_lig = torch.concat([x_lig]*B, dim=0) 
        x_rec = torch.concat([x_rec]*B, dim=0)
        h_lig = torch.concat([h_lig]*B, dim=0) 
        h_rec = torch.concat([h_rec]*B, dim=0)
        prop = torch.concat([prop]*B, dim=0)
        noise_scale_lig = noise_scale[batch_idx_lig][:, None]

        x_lig_noise = (1 - noise_scale_lig) * x_lig + noise_scale_lig * torch.randn_like(x_lig)
        h_lig_noise = (1 - noise_scale_lig) * h_lig + noise_scale_lig * torch.ones_like(h_lig) / self.num_classes
        noise_dist = Categorical(probs=h_lig_noise)
        h_lig_noise = noise_dist.sample()
        h_lig_noise = F.one_hot(h_lig_noise, num_classes=self.num_classes).float()

        noise_prop = (1 - noise_scale) * prop

        atom_scl_out, _, batch_idx = self(x_rec, x_lig_noise, h_rec, h_lig_noise, batch_idx_rec, batch_idx_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)[:, 0]
        loss = torch.abs(final_exp_pred - noise_prop).mean()

        return {'loss': loss}, {'pred': final_exp_pred, 'label': noise_prop}


class AffinityPredictor(nn.Module):
    def __init__(self, cfg, vec_dim=1):
        super().__init__()
        self.cfg = cfg
        # Build the network
        self.num_classes = cfg.get('num_atomtype', 14)
        self.num_layers = cfg['encoder'].get('num_layers', 6)
        self.node_feat_dim = cfg['encoder'].get('node_feat_dim', 128)
        self.vec_feat_dim = cfg['encoder'].get('vec_feat_dim', 128)
        self.edge_feat_dim = cfg['encoder'].get('edge_feat_dim', 4)
        self.num_edge_classes = cfg['encoder'].get('num_edge_classes', None)
        self.fuse_att = MultiHeadAttention(input_dim=self.node_feat_dim, 
                                           hid_dim=256, out_dim=self.node_feat_dim, head_num=4)
        self.cutoff_mode = cfg['encoder'].get('cutoff_mode', 'knn')  # [radius, none]
        self.cut_off = cfg['encoder'].get('k', 48)
        self.r_max = cfg['encoder'].get('r_max', 10.0)
        self.ligand_atom_emb = nn.Linear(self.num_classes, 
                                             cfg.embedder.get('emb_dim', 128))
        self.protein_atom_emb = nn.Linear(len(atomic_numbers)+1, 
                                             cfg.embedder.get('emb_dim', 128))
        self.protein_vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)
        self.ligand_vec_emb = nn.Linear(vec_dim, self.vec_feat_dim)
        self._buid_blocks()

        
        self.classifier = nn.Sequential(
            GVPerceptronVN(self.node_feat_dim,  self.vec_feat_dim , self.node_feat_dim,  self.vec_feat_dim ),
            GVLinear(self.node_feat_dim,  self.vec_feat_dim , 1, 1)
        )
    
    @property
    def out_sca(self):
        return self.node_feat_dim[0]
    
    @property
    def out_vec(self):
        return self.vec_feat_dim[1]
    
    def _buid_blocks(self):
        self.prot_encoder = nn.ModuleList()
        self.lig_encoder = nn.ModuleList()
        for _ in range(self.num_layers):
            block1 = AttentionInteractionBlockVN(
                hidden_channels=(self.node_feat_dim, self.vec_feat_dim),
                edge_channels=self.vec_feat_dim,
                num_edge_types=self.edge_feat_dim + 1,
                r_max = self.r_max
            )
            block2 = AttentionInteractionBlockVN(
                hidden_channels=(self.node_feat_dim, self.vec_feat_dim),
                edge_channels=self.vec_feat_dim,
                num_edge_types=self.edge_feat_dim + 1,
                r_max = self.r_max
            )
            self.prot_encoder.append(block1)
            self.lig_encoder.append(block2)

    def _extend_edge_index(self, x, edge_index, edge_type, batch):
        if self.cutoff_mode == 'radius':
            edge_index_expand = radius_graph(x, r=cut_off, batch=batch, flow='source_to_target')
        elif self.cutoff_mode == 'knn':
            cut_off = int(self.cut_off)
            edge_index_expand = knn_graph(x, k=cut_off, batch=batch, flow='source_to_target')

        edge_type_expand = torch.zeros(edge_index_expand.size(1), dtype=torch.long, device=x.device)

        if edge_index is None:
            edge_index = torch.empty([2,0], dtype=torch.long).to(x.device)

        if edge_type is None:
            edge_type = torch.ones_like(edge_index[0,:]).long()

        if edge_index is not None:
            edge_index = torch.cat([edge_index, edge_index_expand], dim=1)
        
        edge_type = torch.cat([edge_type, edge_type_expand], dim=0)

        edge_index, edge_type = coalesce(edge_index, edge_attr=edge_type, reduce='max') # bond replace knn edge

        return edge_index, edge_type


    def forward(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):
        
        rec_len = x_rec.shape[0]
        lig_len = x_lig.shape[0]

        rec_edge_index, rec_edge_type = self._extend_edge_index(x_rec, None, None, batch_idx_rec)
        rec_edge_attr = F.one_hot(rec_edge_type, num_classes=self.edge_feat_dim + 1).float()
        
        rec_edge_vector = x_rec[rec_edge_index[0]] - x_rec[rec_edge_index[1]]

        lig_edge_index, lig_edge_type = self._extend_edge_index(x_lig, None, None, batch_idx_lig)
        lig_edge_attr = F.one_hot(lig_edge_type, num_classes=self.edge_feat_dim + 1).float()
        
        lig_edge_vector = x_lig[lig_edge_index[0]] - x_lig[lig_edge_index[1]]
        # h_init, vec_init = h.clone(), vec.clone()

        h_lig = self.ligand_atom_emb(h_lig)
        h_rec = self.protein_atom_emb(h_rec)
        vec_rec = self.protein_vec_emb(x_rec.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)
        vec_lig = self.ligand_vec_emb(x_lig.unsqueeze(1).transpose(-1, -2)).transpose(-1, -2)
        for i in range(self.num_layers):
            rec_delta_h, rec_delta_vec = self.prot_encoder[i](h_rec, vec_rec, rec_edge_index, rec_edge_attr, rec_edge_vector)
            lig_delta_h, lig_delta_vec = self.lig_encoder[i](h_lig, vec_lig, lig_edge_index, lig_edge_attr, lig_edge_vector)
            fuse_feat = self.fuse_att(x=torch.concat((rec_delta_h, lig_delta_h), dim=0), 
                                      batch_idx=torch.concat((batch_idx_rec, batch_idx_lig), dim=0))
            h_rec = h_rec + fuse_feat[:rec_len]
            vec_rec = vec_rec + rec_delta_vec
            h_lig = h_lig + fuse_feat[rec_len:]
            vec_lig = vec_lig + lig_delta_vec

        h_out, vec_out = self.classifier((h_lig, vec_lig))
        return h_out, vec_out, batch_idx_lig
    
    def get_loss(self, prop, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):

        h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()
        atom_scl_out, _, batch_idx = self(x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)[:, 0]
        loss = torch.abs(final_exp_pred - prop).mean()

        return {'loss': loss}, {'pred': final_exp_pred, 'label': prop}
    
    def predict(self, x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig):
        if h_lig.ndim == 1:
            h_lig = F.one_hot(h_lig, num_classes=self.num_classes).float()
        atom_scl_out, _, batch_idx = self(x_rec, x_lig, h_rec, h_lig, batch_idx_rec, batch_idx_lig)
        final_exp_pred = scatter_mean(atom_scl_out, batch_idx, dim=0)[:, 0]
        
        return final_exp_pred