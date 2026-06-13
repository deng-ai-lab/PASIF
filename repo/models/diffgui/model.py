from tqdm import tqdm
import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import Adam
from torch_scatter import scatter_sum, scatter_mean
from repo.modules.egnn.transition import ContigousTransition, GeneralCategoricalTransition
from repo.modules.egnn.egnn import EgnnNet
from repo.modules.egnn.common import *
from repo.models.diffgui.diffusion import *
import time
import copy
from repo.models.charge_density.utils import prepare_input_diffgui, prepare_input_given_grid_diffgui

class DiffGui(Module):
    def __init__(self,
        config,
        protein_node_types,
        ligand_node_types,
        num_edge_types,  # explicit bond type: 0, 1, 2, 3, 4
        **kwargs
    ):
        super().__init__()
        self.config = config
        self.protein_node_types = protein_node_types
        self.ligand_node_types = ligand_node_types
        self.num_edge_types = num_edge_types
        self.k = config.knn
        self.cutoff_mode = config.cutoff_mode
        self.center_pos_mode = config.center_pos_mode
        self.bond_len_loss = getattr(config, 'bond_len_loss', False)

        # # define beta and alpha
        self.define_betas_alphas(config.diff)

        # # embedding
        if self.config.node_indicator:
            node_dim = config.node_dim - 1
        else:
            node_dim = config.node_dim
        edge_dim = config.edge_dim
        time_dim = config.diff.time_dim
        class_dim = config.class_dim
        class_emb_dim = config.class_emb_dim
        self.protein_node_embedder = nn.Linear(protein_node_types, node_dim, bias=False) # protein element type
        self.protein_edge_embedder = nn.Linear(num_edge_types, edge_dim, bias=False) # protein bond type
        if self.config.train_mode in ('ori', 'no_bond'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim - class_emb_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim - class_emb_dim, bias=False) # ligand bond type
        elif self.config.train_mode in ('no_lab', 'no_both'):
            self.ligand_node_embedder = nn.Linear(ligand_node_types, node_dim - time_dim, bias=False)  # ligand element type
            self.ligand_edge_embedder = nn.Linear(num_edge_types, edge_dim - time_dim, bias=False) # ligand bond type
        self.time_emb = nn.Sequential(
            GaussianSmearing(stop=self.num_timesteps, num_gaussians=time_dim, type_='linear'),
        )
        self.class_emb = nn.Sequential(
            nn.Linear(class_dim, class_emb_dim * 4),
            nn.LayerNorm(class_emb_dim * 4),
            nn.GELU(),
            nn.Linear(class_emb_dim * 4, class_emb_dim)
        )
        
        # # denoiser
        if config.denoiser.backbone == 'EGNN':
            self.denoiser = EgnnNet(config.node_dim, config.edge_dim, **config.denoiser)
        else:
            raise NotImplementedError(config.denoiser.backbone)

        # # decoder
        self.ligand_node_decoder = MLP(config.node_dim, ligand_node_types, config.node_dim)
        self.ligand_edge_decoder = MLP(config.edge_dim, num_edge_types, config.edge_dim)


    def define_betas_alphas(self, config):
        self.num_timesteps = config.num_timesteps
        self.categorical_space = getattr(config, 'categorical_space', 'discrete')
        
        # try to get the scaling
        if self.categorical_space == 'continuous':
            self.scaling = getattr(config, 'scaling', [1., 1., 1.])
        else:
            self.scaling = [1., 1., 1.]  # actually not used for discrete space (defined for compatibility)

        # # diffusion for pos
        pos_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_pos
        )
        assert self.scaling[0] == 1, 'scaling for pos should be 1'
        self.pos_transition = ContigousTransition(pos_betas)

        # # diffusion for node type
        node_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_atom
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_atom.init_prob
            self.node_transition = GeneralCategoricalTransition(node_betas, self.ligand_node_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_node = self.scaling[1]
            self.node_transition = ContigousTransition(node_betas, self.ligand_node_types, scaling_node)
        else:
            raise ValueError(self.categorical_space)

        # # diffusion for edge type
        edge_betas = get_beta_schedule(
            num_timesteps=self.num_timesteps,
            **config.diff_bond
        )
        if self.categorical_space == 'discrete':
            init_prob = config.diff_bond.init_prob
            self.edge_transition = GeneralCategoricalTransition(edge_betas, self.num_edge_types,
                                                            init_prob=init_prob)
        elif self.categorical_space == 'continuous':
            scaling_edge = self.scaling[2]
            self.edge_transition = ContigousTransition(edge_betas, self.num_edge_types, scaling_edge)
        else:
            raise ValueError(self.categorical_space)

    def sample_time(self, num_graphs, device, **kwargs):
        time_step = torch.randint(
            0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
        time_step = torch.cat(
            [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
        pt = torch.ones_like(time_step).float() / self.num_timesteps
        return time_step, pt
    
    def fix_zero_time(self, num_graphs, device, **kwargs): 
        time_step = torch.zeros(num_graphs, dtype=torch.long, device=device)   
        pt = torch.ones_like(time_step).float() / self.num_timesteps  
        return time_step, pt

    def _get_edge_index(self, x, batch, ligand_mask):
        if self.cutoff_mode == "knn":
            edge_index = knn_graph(x, k=self.k, batch=batch, flow="target_to_source")
        elif self.cutoff_mode == "hybrid":
            edge_index = batch_hybrid_edge_connection(
                x, k=self.k, ligand_mask=ligand_mask, batch=batch, add_p_index=True
            )
        else:
            raise ValueError(
                f"Unsupported cutoff mode: {self.cutoff_mode}! Please select cutoff mode among knn, hybrid."
            )
        return edge_index

    def _get_edge_type(self, edge_index, ligand_mask):
        src, dst = edge_index
        edge_type = torch.zeros(len(src), dtype=torch.int64).to(edge_index.device)
        n_src = ligand_mask[src] == 1
        n_dst = ligand_mask[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3

        nonzero_indices = torch.nonzero(edge_type).flatten()
        edge_type = torch.index_select(edge_type, dim=0, index=nonzero_indices)
        edge_type = torch.zeros_like(edge_type)
        edge_index = torch.index_select(edge_index, dim=1, index=nonzero_indices)
        edge_type = F.one_hot(edge_type, num_classes=self.num_edge_types)
        return edge_type, edge_index

    def forward(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node_pert, ligand_pos_pert, ligand_batch,
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        t, lab
    ):
        """
        Predict Ligand at step `0` given perturbed Ligand at step `t` with hidden dims and time step
        """
        # 1 node, edge and time embedding
        time_embed_node = self.time_emb(t.index_select(0, ligand_batch))
        class_embed_node = self.class_emb(lab.index_select(0, ligand_batch))
        time_embed_edge = self.time_emb(t.index_select(0, ligand_edge_batch))
        class_embed_edge = self.class_emb(lab.index_select(0, ligand_edge_batch))
        if self.config.train_mode in ('ori', 'no_bond'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node, class_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge, class_embed_edge], dim=-1)
        elif self.config.train_mode in ('no_lab', 'no_both'):
            ligand_node_h_pert = torch.cat([self.ligand_node_embedder(ligand_node_pert), time_embed_node], dim=-1)
            ligand_edge_h_pert = torch.cat([self.ligand_edge_embedder(ligand_edge_pert), time_embed_edge], dim=-1)
        protein_h = self.protein_node_embedder(protein_node)

        if self.config.node_indicator:
            protein_h = torch.cat([protein_h, torch.zeros(len(protein_h), 1).to(protein_h)], -1)
            ligand_node_h_pert = torch.cat([ligand_node_h_pert, torch.ones(len(ligand_node_h_pert), 1).to(ligand_node_h_pert)], -1)

        # 2 combine protein and ligand input
        all_node_h, all_node_pos, all_node_batch, ligand_mask = compose(
            protein_h, protein_pos, protein_batch, ligand_node_h_pert, ligand_pos_pert, ligand_batch
        )

        sub_edge_index = self._get_edge_index(all_node_pos, all_node_batch, ligand_mask)
        sub_edge_type, sub_edge_index = self._get_edge_type(sub_edge_index, ligand_mask)
        sub_edge_batch = all_node_batch[sub_edge_index[0]]
        sub_edge_h = self.protein_edge_embedder(sub_edge_type.to(torch.float32))
        node_batch_counts = torch.bincount(all_node_batch)
        ligand_node_batch_counts = torch.bincount(ligand_batch)
        cumulative_nodes = torch.cat([torch.tensor([0]).to(all_node_batch.device), torch.cumsum(node_batch_counts, dim=0)[:-1]])
        cumulative_ligand_nodes = torch.cat([torch.tensor([0]).to(ligand_batch.device), torch.cumsum(ligand_node_batch_counts, dim=0)[:-1]])
        new_ligand_edge_index = ligand_edge_index + cumulative_nodes[ligand_edge_batch] - cumulative_ligand_nodes[ligand_edge_batch]
        all_edge_h, all_edge_index, all_edge_batch, ligand_edge_mask = edge_compose(
            sub_edge_h, sub_edge_index, sub_edge_batch, ligand_edge_h_pert, new_ligand_edge_index, ligand_edge_batch
        )

        # 3 diffuse to get the updated node embedding and bond embedding
        node_h, node_pos, edge_h = self.denoiser(
            node_h=all_node_h,
            node_pos=all_node_pos, 
            edge_h=all_edge_h, 
            edge_index=all_edge_index,
            node_time=t.index_select(0, all_node_batch).unsqueeze(-1) / self.num_timesteps,
            edge_time=t.index_select(0, all_edge_batch).unsqueeze(-1) / self.num_timesteps,
            ligand_mask=ligand_mask
        )
        
        ligand_node_h = node_h[ligand_mask]
        ligand_node_pos = node_pos[ligand_mask]
        ligand_edge_h = edge_h[ligand_edge_mask]
        n_halfedges = ligand_edge_h.shape[0] // 2
        pred_ligand_node = self.ligand_node_decoder(ligand_node_h)
        pred_ligand_halfedge = self.ligand_edge_decoder(ligand_edge_h[:n_halfedges] + ligand_edge_h[n_halfedges:])
        pred_ligand_pos = ligand_node_pos
        
        return {
            'pred_ligand_node': pred_ligand_node,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': pred_ligand_halfedge
        }  # ligand at step 0

    def get_loss(
        self, protein_node, protein_pos, protein_batch, 
        ligand_node, ligand_pos, ligand_batch,
        halfedge_type, halfedge_index, halfedge_batch,
        num_mol, batch_lab
    ):
        num_graphs = num_mol
        device = ligand_pos.device
        protein_pos, ligand_pos, _ = center_pos(
            protein_pos, ligand_pos, protein_batch, ligand_batch, mode=self.center_pos_mode
        )

        # 1. sample noise levels
        time_step, _ = self.sample_time(num_graphs, device)

        # 2. perturb pos, node, edge
        pos_pert = self.pos_transition.add_noise(ligand_pos, time_step, ligand_batch)
        node_pert = self.node_transition.add_noise(ligand_node, time_step, ligand_batch)
        halfedge_pert = self.edge_transition.add_noise(halfedge_type, time_step, halfedge_batch)
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)  # undirected edges
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        if self.categorical_space == 'discrete':
            ligand_node_pert, log_node_t, log_node_0 = node_pert
            ligand_halfedge_pert, log_halfedge_t, log_halfedge_0 = halfedge_pert
        else:
            ligand_node_pert, ligand_node_0 = node_pert
            ligand_halfedge_pert, ligand_halfedge_0 = halfedge_pert
        
        ligand_edge_pert = torch.cat([ligand_halfedge_pert, ligand_halfedge_pert], dim=0)
        ligand_pos_pert = pos_pert

        # 3. forward to denoise
        preds = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )
        pred_ligand_node = preds['pred_ligand_node']
        pred_ligand_pos = preds['pred_ligand_pos']
        pred_ligand_halfedge = preds['pred_ligand_halfedge']

        # 4. loss
        # 4.1 pos loss
        loss_pos = F.mse_loss(pred_ligand_pos, ligand_pos)
        if self.bond_len_loss == True:
            bond_index = halfedge_index[:, halfedge_type > 0]
            true_length = torch.norm(ligand_pos[bond_index[0]] - ligand_pos[bond_index[1]], dim=-1)
            pred_length = torch.norm(pred_ligand_pos[bond_index[0]] - pred_ligand_pos[bond_index[1]], dim=-1)
            loss_len = F.mse_loss(pred_length, true_length)
    
        if self.categorical_space == 'discrete':
            # 4.2 node type loss
            log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
            log_node_post_true = self.node_transition.q_v_posterior(log_node_0, log_node_t, time_step, ligand_batch, v0_prob=True)
            log_node_post_pred = self.node_transition.q_v_posterior(log_node_recon, log_node_t, time_step, ligand_batch, v0_prob=True)
            kl_node = self.node_transition.compute_v_Lt(log_node_post_true, log_node_post_pred, log_node_0, t=time_step, batch=ligand_batch)
            loss_node = torch.mean(kl_node) * 100
            # 4.3 edge type loss
            log_halfedge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
            log_edge_post_true = self.edge_transition.q_v_posterior(log_halfedge_0, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            log_edge_post_pred = self.edge_transition.q_v_posterior(log_halfedge_recon, log_halfedge_t, time_step, halfedge_batch, v0_prob=True)
            kl_edge = self.edge_transition.compute_v_Lt(log_edge_post_true, log_edge_post_pred, log_halfedge_0, t=time_step, batch=halfedge_batch)
            loss_edge = torch.mean(kl_edge)  * 100
        else:
            loss_node = F.mse_loss(pred_ligand_node, ligand_node_0)  * 30
            loss_edge = F.mse_loss(pred_ligand_halfedge, ligand_halfedge_0) * 30

        # total loss
        if self.config.train_mode in ('ori', 'no_lab'):
            loss_total = loss_pos + loss_node + loss_edge + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
            'loss': loss_total,
            'loss_pos': loss_pos,
            'loss_node': loss_node,
            'loss_edge': loss_edge
        }
        elif self.config.train_mode in ('no_bond', 'no_both'):
            loss_total = loss_pos + loss_node + (loss_len if self.bond_len_loss else 0)
            loss_dict = {
                'loss': loss_total,
                'loss_pos': loss_pos,
                'loss_node': loss_node
            }
        if self.bond_len_loss == True:
            loss_dict['loss_len'] = loss_len

        pred_dict = {
            'pred_ligand_node': F.softmax(pred_ligand_node, dim=-1),
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_halfedge': F.softmax(pred_ligand_halfedge, dim=-1)
        }
        return loss_dict, pred_dict

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_eps = extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - \
                      extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch) * eps
        return pos0_from_eps

    def _predict_eps_from_x0(self, xt, t, pred_x0, batch):
        return (
            (extract(self.pos_transition.sqrt_recip_alphas_bar, t, batch) * xt - pred_x0) /
            extract(self.pos_transition.sqrt_recipm1_alphas_bar, t, batch)
        )

    def classifier_free(
        self, protein_node, protein_pos, protein_batch,
        ligand_node_pert, ligand_pos_pert, ligand_batch, 
        ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
        gui_strength, time_step, batch_lab
    ):
        """
        Compute new results for the start step in classifier free diffusion sampling.
        """
        # t0 = time.time()
        preds_cond = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )
        # t1 = time.time()

        batch_lab_zero = torch.zeros(batch_lab.shape, device=ligand_batch.device)
        # t2 = time.time()
        preds_uncond = self(
            protein_node, protein_pos, protein_batch,
            ligand_node_pert, ligand_pos_pert, ligand_batch,
            ligand_edge_pert, ligand_edge_index, ligand_edge_batch, 
            time_step, batch_lab
        )
        # t3 = time.time()

        pred_eps_cond = self._predict_eps_from_x0(
            xt=ligand_pos_pert, t=time_step, pred_x0=preds_cond['pred_ligand_pos'], batch=ligand_batch
        )
        pred_eps_uncond = self._predict_eps_from_x0(
            xt=ligand_pos_pert, t=time_step, pred_x0=preds_uncond['pred_ligand_pos'], batch=ligand_batch
        )
        pred_eps = (1 + gui_strength) * pred_eps_cond - gui_strength * pred_eps_uncond
        pred_ligand_pos = self._predict_x0_from_eps(xt=ligand_pos_pert, t=time_step, eps=pred_eps, batch=ligand_batch)

        pred_ligand_node = preds_cond['pred_ligand_node'] + preds_uncond['pred_ligand_node']
        pred_ligand_halfedge = preds_cond['pred_ligand_halfedge'] + preds_uncond['pred_ligand_halfedge']

        return pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge

    @torch.no_grad()
    def sample(
        self, n_graphs, 
        protein_node, protein_pos, protein_batch, 
        ligand_batch, halfedge_index, halfedge_batch, 
        batch_lab=None, gui_strength=None, 
        bond_predictor=None, guidance=None
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_all = len(ligand_batch)
        n_halfedges_all = len(halfedge_batch)
        
        node_init = self.node_transition.sample_init(n_nodes_all)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_all)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init
            
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)

        # # 1.1 log init
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_all, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_all, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            # # 2.1 inference
            if self.config.train_mode in ('ori', 'no_bond'):
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = self.classifier_free(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch, 
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    gui_strength, time_step, batch_lab
                )
            elif self.config.train_mode in ('no_lab', 'no_both'):
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']
            
            # # 2.2 get the t - 1 state
            # pos 
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.3 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # 2.4 log update
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.5 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch] 
        # # 3. get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    @torch.no_grad()
    def sample_frag(
        self, n_graphs, 
        protein_node, protein_pos, protein_batch, 
        frag_node, frag_pos, frag_batch,
        frag_halfedge_type, frag_halfedge_index, frag_halfedge_batch,
        ligand_batch, halfedge_index, halfedge_batch, 
        batch_lab=None, gui_strength=None, 
        bond_predictor=None, guidance=None, gen_mode=None
    ):
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        frag_node_mask = get_fragment_mask(ligand_batch, frag_batch)
        frag_halfedge_mask = get_fragment_mask(halfedge_batch, frag_halfedge_batch)
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)
        frag_pos = frag_pos - offset[frag_batch]

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            # # 2.1 get the init values for fragment
            if gen_mode == 'frag_cond':
                frag_pos_pert = frag_pos
                log_frag_node_type = index_to_log_onehot(frag_node, self.ligand_node_types)
                frag_node_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_node, self.ligand_node_types)), self.ligand_node_types).float()
                log_frag_halfedge_type = index_to_log_onehot(frag_halfedge_type, self.num_edge_types)
                frag_halfedge_pert = F.one_hot(log_sample_categorical(index_to_log_onehot(frag_halfedge_type, self.num_edge_types)), self.num_edge_types).float()
            elif gen_mode == 'frag_diff':
                pos_pert = self.pos_transition.add_noise(frag_pos, time_step, frag_batch)
                node_pert = self.node_transition.add_noise(frag_node, time_step, frag_batch)
                halfedge_pert = self.edge_transition.add_noise(frag_halfedge_type, time_step, frag_halfedge_batch)
                
                if self.categorical_space == 'discrete':
                    frag_node_pert, log_frag_node_type, _ = node_pert
                    frag_halfedge_pert, log_frag_halfedge_type, _ = halfedge_pert
                else:
                    frag_node_pert, _ = node_pert
                    frag_halfedge_pert, _ = halfedge_pert
                frag_pos_pert = pos_pert

            # # 2.2 combine fragment and ligand
            ligand_pos_pert[frag_node_mask] = frag_pos_pert
            ligand_node_h_pert[frag_node_mask] = frag_node_pert
            ligand_halfedge_h_pert[frag_halfedge_mask] = frag_halfedge_pert
            log_node_type[frag_node_mask] = log_frag_node_type
            log_halfedge_type[frag_halfedge_mask] = log_frag_halfedge_type
            ligand_edge_h_pert = torch.cat([ligand_halfedge_h_pert, ligand_halfedge_h_pert], dim=0)
            
            # # 2.3 inference
            if self.config.train_mode in ('ori', 'no_bond'):
                # t0 = time.time()
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = self.classifier_free(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch, 
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    gui_strength, time_step, batch_lab
                )
                # t1 = time.time()
            elif self.config.train_mode in ('no_lab', 'no_both'):
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    ligand_node_h_pert, ligand_pos_pert, ligand_batch,
                    ligand_edge_h_pert, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

            # # 2.4 get the t - 1 state
            # pos 
            ligand_pos_prev = self.pos_transition.get_prev_from_recon(
                x_t=ligand_pos_pert, x_recon=pred_ligand_pos, t=time_step, batch=ligand_batch
            )
            if self.categorical_space == 'discrete':
                # node types
                log_node_recon = F.log_softmax(pred_ligand_node, dim=-1)
                log_node_type = self.node_transition.q_v_posterior(log_node_recon, log_node_type, time_step, ligand_batch, v0_prob=True)
                ligand_node_type_prev = log_sample_categorical(log_node_type)
                ligand_node_h_prev = self.node_transition.onehot_encode(ligand_node_type_prev)
                
                # halfedge types
                log_edge_recon = F.log_softmax(pred_ligand_halfedge, dim=-1)
                log_halfedge_type = self.edge_transition.q_v_posterior(log_edge_recon, log_halfedge_type, time_step, halfedge_batch, v0_prob=True)
                ligand_halfedge_type_prev = log_sample_categorical(log_halfedge_type)
                ligand_halfedge_h_prev = self.edge_transition.onehot_encode(ligand_halfedge_type_prev)
                
            else:
                ligand_node_h_prev = self.node_transition.get_prev_from_recon(
                    x_t=ligand_node_h_pert, x_recon=pred_ligand_node, t=time_step, batch=ligand_batch)
                ligand_halfedge_h_prev = self.edge_transition.get_prev_from_recon(
                    x_t=ligand_halfedge_h_pert, x_recon=pred_ligand_halfedge, t=time_step, batch=halfedge_batch)

            # # 2.5 use guidance to modify pos
            if self.config.train_mode not in ('no_bond', 'no_both'):
                if guidance is not None:
                    gui_type, gui_scale = guidance
                    if (gui_scale > 0):
                        with torch.enable_grad():
                            ligand_node_h_in = ligand_node_h_pert.detach()
                            ligand_pos_in = ligand_pos_pert.detach().requires_grad_(True)
                            
                            # t0 = time.time()
                            pred_bondpredictor = bond_predictor(
                                protein_node, protein_pos, protein_batch,
                                ligand_node_h_in, ligand_pos_in, ligand_batch,
                                ligand_edge_index, ligand_edge_batch, time_step)
                            # t1 = time.time()
                            delta = self.bond_guidance(gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, ligand_halfedge_type_prev, log_halfedge_type)
                        ligand_pos_prev = ligand_pos_prev + delta

            # 2.6 update trajectory
            ligand_node_traj[i+1] = ligand_node_h_prev
            ligand_pos_traj[i+1] = ligand_pos_prev + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = ligand_halfedge_h_prev

            # # 2.7 update t-1
            ligand_pos_pert = ligand_pos_prev
            ligand_node_h_pert = ligand_node_h_prev
            ligand_halfedge_h_pert = ligand_halfedge_h_prev

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]

        # # 3. get the final positions
        return {
            'pred': [pred_ligand_node, pred_ligand_pos, pred_ligand_halfedge],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }
    
    def loss_noise_p0t(self, x_t, t, pt0, q0, p0t, batch_idx):
        """
        param:
            -x_t: (L, S)
            -t: (N, )
            -pt0: (L, S, S)
            -q0: (L, S)
            -p0t: (L, S)
            -batch_idx: (L, )
        """
        # L = batch_idx.shape[0]
        L, S = x_t.shape
        x_t_idx = torch.argmax(x_t, dim=-1) # (L,)

        pt0_x = pt0[torch.arange(L), :, x_t_idx]  # (L, S)

        qt = torch.einsum('...i,...ij->...j', q0, pt0)
        qt_x = qt[torch.arange(L), x_t_idx][:, None]  # (L, 1)

        q0t = (q0[:, :, None] * pt0) / qt[:, None, :]  # (L, S, S)
        q0t_x = q0t[torch.arange(L), :, x_t_idx]  # (L, S)

        grad = pt0_x.clamp(min=1.e-3, max=0.999) * torch.log(q0t_x.clamp(min=1.e-3, max=0.999) / p0t.clamp(min=1.e-3, max=0.999)) / qt_x.clamp(min=1.e-3, max=0.999)

        loss_out = (grad.detach() * q0).mean()

        return loss_out, grad

    def loss_noise_pt(self, x_t, t, pT0, pt0, q0, p0t, batch_idx):
        """
        param:
            -x_t: (L, S)
            -t: (N, )
            -q0: (L, S)
            -p0t: (L, S)
            -pT0: (L, S, S)
            -pt0: (L, S, S)
            -batch_idx: (L, )
        """
        L, S = x_t.shape
        x_t_idx = torch.argmax(x_t, dim=-1)

        pT0_x = pT0[torch.arange(L), :, x_t_idx]
        pt0_x = pt0[torch.arange(L), :, x_t_idx]

        pT_pt = p0t * pT0_x / pt0_x.clamp(min=0.001, max=0.999)
        pT_pt = (pT_pt).sum(-1)

        pT_pt = pT_pt[:, None] + 1.e-5

        qt = torch.einsum('...i,...ij->...j', q0, pt0)
        qT = torch.einsum('...i,...ij->...j', q0, pT0)
        qt_x = qt[torch.arange(L), x_t_idx]  # (L,)
        qT_x = qT[torch.arange(L), x_t_idx]
        qt_qT = qt_x / qT_x
        qt_qT = qt_qT[:, None] + 1.e-5

        grad = pt0[torch.arange(L), :, x_t_idx] * (torch.log(qt_qT) + torch.log(pT_pt)) / qt_x[:, None] + 1.  # (L, S)

        loss_out = (grad.detach() * q0).mean()

        return loss_out, grad
    
    def softmax_jacobian(self, z: torch.Tensor, temp) -> torch.Tensor:
        s = F.softmax(z/temp, dim=-1)
        B, N = s.shape
        # diag(s) -> (B, N, N)
        diag_s = torch.diag_embed(s)

        # outer product s s^T -> (B, N, N)
        outer_s = s.unsqueeze(2) @ s.unsqueeze(1)

        return (diag_s - outer_s) / temp
    
    def dr_slover(
    self, n_graphs, 
    protein_node, protein_pos, protein_batch, 
    frag_node, frag_pos, frag_batch,
    frag_halfedge_type, frag_halfedge_index, frag_halfedge_batch,
    ligand_batch, halfedge_index, halfedge_batch, 
    batch_lab=None, gui_strength=None, 
    bond_predictor=None, guidance=None, gen_mode=None
    ):
        # # ligand type: last dim is absort state  
        # # edge type  : first dim is absort state
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        frag_node_mask = get_fragment_mask(ligand_batch, frag_batch)
        frag_halfedge_mask = get_fragment_mask(halfedge_batch, frag_halfedge_batch)
        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)
        frag_pos = frag_pos - offset[frag_batch]

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        
        sigma_x0  = 0.0001
        sigma_c0  = 0.001
        sigma_he0 = 0.001
        c_init_prob  = torch.tensor(self.node_transition.init_prob, device=frag_pos.device).float()
        he_init_prob = torch.tensor(self.edge_transition.init_prob, device=frag_pos.device).float()

        x_lig_in  = ligand_pos_pert.detach().clone()
        c_lig_in  = ligand_node_h_pert.detach().clone()
        he_lig_in = ligand_halfedge_h_pert.detach().clone()

        x_lig_in[frag_node_mask]      = frag_pos
        c_lig_in[frag_node_mask]      = index_to_log_onehot(frag_node, self.ligand_node_types).exp().detach().clone()
        he_lig_in[frag_halfedge_mask] = index_to_log_onehot(frag_halfedge_type, self.num_edge_types).exp().detach().clone()

        x_lig_mu  = torch.autograd.Variable(x_lig_in.detach().clone(), requires_grad=True).float()
        c_lig_mu  = torch.autograd.Variable(c_lig_in.detach().clone(), requires_grad=True).float()
        he_lig_mu = torch.autograd.Variable(he_lig_in.detach().clone(), requires_grad=True).float()
        optimizer = Adam(params=[x_lig_mu, c_lig_mu, he_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
        h_mask_node = frag_node_mask[:, None].float()
        h_mask_he = frag_halfedge_mask[:, None].float()
        temp = 0.5
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            if i == 0:
                c_lig_mu_norm  = c_lig_mu
                he_lig_mu_norm = he_lig_mu
            else:
                c_lig_mu_norm  = F.softmax(c_lig_mu, dim=-1)
                he_lig_mu_norm = F.softmax(he_lig_mu, dim=-1)

            x0_pred  = x_lig_mu + sigma_x0 * torch.randn_like(x_lig_mu)
            c0_pred  = c_lig_mu_norm
            he0_pred = he_lig_mu_norm
            dist_c   = Categorical(probs=c0_pred)
            dist_he  = Categorical(probs=he0_pred)
            cv_pred  = dist_c.sample()
            hev_pred = dist_he.sample()

            x_lig_t  = self.pos_transition.add_noise(x0_pred, time_step, ligand_batch)
            c_lig_t, _, _  = self.node_transition.add_noise(cv_pred, time_step, ligand_batch)
            he_lig_t, _, _ = self.edge_transition.add_noise(hev_pred, time_step, halfedge_batch)
            e_lig_t  = torch.cat([he_lig_t, he_lig_t], dim=0)       
            
            # # 2.3 inference
            with torch.no_grad():
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    c_lig_t, x_lig_t, ligand_batch,
                    e_lig_t, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

                pred_x_0  = pred_ligand_pos
                pred_c_0  = F.softmax(pred_ligand_node, dim=-1)
                pred_he_0 = F.softmax(pred_ligand_halfedge, dim=-1)

            node_pt0 = self.node_transition.q_mats[step][None, ...].repeat(n_nodes_ligand, 1, 1)
            node_pT0 = self.node_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_nodes_ligand, 1, 1)
            he_pt0 = self.edge_transition.q_mats[step][None, ...].repeat(n_halfedges_ligand, 1, 1)
            he_pT0 = self.edge_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_halfedges_ligand, 1, 1)
            pyx_c  = (1 - sigma_c0) * c_lig_in + sigma_c0 * c_init_prob
            pyx_he = (1 - sigma_he0) * he_lig_in + sigma_he0 * he_init_prob

            grad_x_obs = h_mask_node * (x_lig_mu - x_lig_in)
            grad_x_noise = (x_lig_mu - pred_ligand_pos).detach()
            grad_c_obs = - h_mask_node * (torch.log(pyx_c.clamp(min=1.e-3, max=0.999)))
            grad_he_obs = - h_mask_he * (torch.log(pyx_he.clamp(min=1.e-3, max=0.999)))
            loss_noise_p0t, grad_c_p0t = self.loss_noise_p0t(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, batch_idx=frag_batch)
            loss_noise_pt, grad_c_pt   = self.loss_noise_pt(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, pT0=node_pT0, batch_idx=frag_batch)
            loss_noise_p0t, grad_he_p0t = self.loss_noise_p0t(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, batch_idx=frag_halfedge_batch)
            loss_noise_pt, grad_he_pt   = self.loss_noise_pt(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, pT0=he_pT0, batch_idx=frag_halfedge_batch)

            grad_c_noise = grad_c_p0t + grad_c_pt
            grad_he_noise = grad_he_p0t + grad_he_pt
            grad_c_norm = grad_c_obs + 3.0 * grad_c_noise
            grad_he_norm = grad_he_obs + 3.0 * grad_he_noise

            if torch.isnan(grad_c_noise).any().item() == True or torch.isinf(grad_c_noise).any().item() == True:
                breakpoint()
            if torch.isnan(grad_he_noise).any().item() == True or torch.isinf(grad_he_noise).any().item() == True:
                breakpoint()

            grad_x = grad_x_obs + 0.25 * grad_x_noise
            jacob = self.softmax_jacobian(c_lig_mu, temp)
            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]
            jacob = self.softmax_jacobian(he_lig_mu, temp)
            grad_he = torch.matmul(jacob.transpose(-1, -2), grad_he_norm[:, :, None])[:, :, 0]

            optimizer.zero_grad()
            x_lig_mu.grad  = grad_x
            c_lig_mu.grad  = grad_c
            he_lig_mu.grad = grad_he
            optimizer.step()

            # 2.6 update trajectory
            ligand_node_traj[i+1] = F.softmax(c_lig_mu/temp, dim=-1).detach()
            ligand_pos_traj[i+1] = x_lig_mu.detach() + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = F.softmax(he_lig_mu/temp, dim=-1).detach()

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]

        # # 3. get the final positions
        return {
            'pred': [ligand_node_traj[self.num_timesteps].log(), ligand_pos_traj[self.num_timesteps], ligand_halfedge_traj[self.num_timesteps].log()],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }
    
    def charge_density_guid(
    self, n_graphs, 
    protein_node, protein_pos, protein_batch, 
    ligand_batch, halfedge_index, halfedge_batch, 
    classifier, params,
    just_lig=False,
    batch_lab=None, gui_strength=None, 
    bond_predictor=None, guidance=None, gen_mode=None,
    local=False
    ):
        # # ligand type: last dim is absort state  
        # # edge type  : first dim is absort state
        B = protein_batch.max() + 1
        device = ligand_batch.device
        tar_prop = params['tar_density'][None, :].repeat(B, 1)
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)

        tar_prop = params['tar_density'][None, :].repeat(B, 1)
        gird_pos = params['grid_pos'] - offset[:1].cpu().numpy()
        if local:
            tar_mask = params['charge_mask']
            charge_mask = tar_mask.reshape(100, 100, 100)
            charge_mask = charge_mask[::2, ::2, ::2].reshape(-1).bool()
            gird_pos = torch.tensor(gird_pos, device=charge_mask.device)
            gird_pos = gird_pos.reshape(100, 100, 100, 3)
            gird_pos = gird_pos[::2, ::2, ::2, :].reshape(-1, 3)
            gird_pos = gird_pos[charge_mask]
            gird_pos = gird_pos.cpu().numpy()
            tar_prop = params['tar_density'].reshape(100, 100, 100)
            tar_prop = tar_prop[::2, ::2, ::2].reshape(-1)
            tar_prop = tar_prop[charge_mask]
            tar_prop = tar_prop[None, :].repeat(B, 1)

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        
        sigma_x0  = 0.0001

        x_lig_in  = ligand_pos_pert.detach().clone()
        c_lig_in  = ligand_node_h_pert.detach().clone()
        he_lig_in = ligand_halfedge_h_pert.detach().clone()

        x_lig_mu  = torch.autograd.Variable(x_lig_in.detach().clone(), requires_grad=True).float()
        c_lig_mu  = torch.autograd.Variable(c_lig_in.detach().clone(), requires_grad=True).float()
        he_lig_mu = torch.autograd.Variable(he_lig_in.detach().clone(), requires_grad=True).float()
        optimizer = Adam(params=[x_lig_mu, c_lig_mu, he_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
        temp = 0.5
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            if i == 0:
                c_lig_mu_norm  = c_lig_mu
                he_lig_mu_norm = he_lig_mu
            else:
                c_lig_mu_norm  = F.softmax(c_lig_mu, dim=-1)
                he_lig_mu_norm = F.softmax(he_lig_mu, dim=-1)

            x0_pred  = x_lig_mu + sigma_x0 * torch.randn_like(x_lig_mu)
            c0_pred  = c_lig_mu_norm
            he0_pred = he_lig_mu_norm
            dist_c   = Categorical(probs=c0_pred)
            dist_he  = Categorical(probs=he0_pred)
            cv_pred  = dist_c.sample()
            hev_pred = dist_he.sample()

            x_lig_t  = self.pos_transition.add_noise(x0_pred, time_step, ligand_batch)
            c_lig_t, _, _  = self.node_transition.add_noise(cv_pred, time_step, ligand_batch)
            he_lig_t, _, _ = self.edge_transition.add_noise(hev_pred, time_step, halfedge_batch)
            e_lig_t  = torch.cat([he_lig_t, he_lig_t], dim=0)       
            
            # # 2.3 inference
            with torch.no_grad():
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    c_lig_t, x_lig_t, ligand_batch,
                    e_lig_t, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

                pred_c_0  = F.softmax(pred_ligand_node, dim=-1)
                pred_he_0 = F.softmax(pred_ligand_halfedge, dim=-1)
            
            if i >= 600:
                
                input_dict = copy.deepcopy(params)
                input_dict['grid_pos'] = gird_pos
                outputs = {'pred':(c_lig_mu_norm.log(), x_lig_mu, he_lig_mu_norm.log()),
                           'traj':(c_lig_mu_norm[None, ...], x_lig_mu[None, ...], he_lig_mu_norm[None, ...]),}
                outputs = {key:[v.detach().cpu().numpy() for v in value] for key, value in outputs.items()}
                cls_batch = prepare_input_given_grid_diffgui(
                    outputs=outputs,
                    batch_node=ligand_batch.detach().cpu().numpy(), 
                    halfedge_index=halfedge_index.detach().cpu().numpy(), 
                    batch_halfedge=halfedge_batch.detach().cpu().numpy(), 
                    label=tar_prop.cpu().numpy(),
                    offset=offset.cpu().numpy(),
                    device=device, params=input_dict, 
                    local=local
                )
                cls_batch['atom_xyz'].requires_grad = True
                prop_pred = classifier(cls_batch)

                prop_pred = prop_pred.clamp(min=0.0)
                tar_rho = cls_batch['label']
                
                if local is False:
                    tar_rho = tar_rho / torch.norm(tar_rho, dim=-1, keepdim=True)
                    prop_pred = prop_pred / torch.norm(prop_pred, dim=-1, keepdim=True)
                    high_rho = tar_rho.detach().clone()
                    peak_rho = tar_rho.detach().clone()
                    high_rho[high_rho < 0.005] = -0.02
                    high_rho[high_rho >= 0.005] = 0.0
                    peak_rho[peak_rho < 0.05] = 0.0
                    peak_rho[peak_rho >= 0.05] = 0.02
                    loss_high =  - (high_rho * prop_pred).sum(-1).mean(0)
                    loss_peak = - (peak_rho * prop_pred).sum(-1).mean(0)
                    loss_prop = loss_high + 0.5 * loss_peak

                    w = 10.0 * i/1000 + 0.0 * (1 - i/1000)

                else:
                    scale = tar_rho.sum(-1, keepdim=True) / prop_pred.sum(-1, keepdim=True)
                    prop_pred = scale * prop_pred
                    tar_rho = tar_rho.clamp(min=1.e-3, max=10)
                    prop_pred = prop_pred.clamp(min=1.e-3, max=10)
                    loss_shape = (prop_pred - tar_rho).abs().sum(-1) / tar_rho.abs().sum(-1)
                    loss_shape = loss_shape.mean()
                    loss_scale = (scale - 1.).abs().mean()
                    loss_prop = loss_shape + 0. * loss_scale

                    w = 20.0 * i/1000 + 0.0 * (1 - i/1000)
                
                grad_prop_x = torch.autograd.grad(loss_prop, cls_batch['atom_xyz'], retain_graph=True)[0] * B
                grad_prop_x = grad_prop_x.reshape(-1, 3)[cls_batch['heavy_mask'].reshape(-1)]
                grad_prop_c = torch.zeros_like(c_lig_mu_norm)
                if torch.isnan(grad_prop_x).any().item() == True:
                    breakpoint()
            else:
                prop_pred = None
                grad_prop_x = torch.zeros_like(x_lig_mu)
                grad_prop_c = torch.zeros_like(c_lig_mu_norm)
                w = 0.0

            node_pt0 = self.node_transition.q_mats[step][None, ...].repeat(n_nodes_ligand, 1, 1)
            node_pT0 = self.node_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_nodes_ligand, 1, 1)
            he_pt0 = self.edge_transition.q_mats[step][None, ...].repeat(n_halfedges_ligand, 1, 1)
            he_pT0 = self.edge_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_halfedges_ligand, 1, 1)

            grad_x_noise = (x_lig_mu - pred_ligand_pos).detach()
            loss_noise_p0t, grad_c_p0t = self.loss_noise_p0t(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, batch_idx=ligand_batch)
            loss_noise_pt, grad_c_pt   = self.loss_noise_pt(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, pT0=node_pT0, batch_idx=ligand_batch)
            loss_noise_p0t, grad_he_p0t = self.loss_noise_p0t(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, batch_idx=halfedge_batch)
            loss_noise_pt, grad_he_pt   = self.loss_noise_pt(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, pT0=he_pT0, batch_idx=halfedge_batch)

            grad_c_noise = grad_c_p0t + grad_c_pt
            grad_he_noise = grad_he_p0t + grad_he_pt
            grad_c_norm = 3.0 * grad_c_noise  + w * grad_prop_c
            grad_he_norm = 3.0 * grad_he_noise

            if torch.isnan(grad_c_noise).any().item() == True or torch.isinf(grad_c_noise).any().item() == True:
                breakpoint()
            if torch.isnan(grad_he_noise).any().item() == True or torch.isinf(grad_he_noise).any().item() == True:
                breakpoint()

            grad_x = 0.25 * grad_x_noise + w * grad_prop_x
            jacob = self.softmax_jacobian(c_lig_mu, temp)
            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]
            jacob = self.softmax_jacobian(he_lig_mu, temp)
            grad_he = torch.matmul(jacob.transpose(-1, -2), grad_he_norm[:, :, None])[:, :, 0]

            optimizer.zero_grad()
            x_lig_mu.grad  = grad_x
            c_lig_mu.grad  = grad_c
            he_lig_mu.grad = grad_he
            optimizer.step()

            # 2.6 update trajectory
            ligand_node_traj[i+1] = F.softmax(c_lig_mu/temp, dim=-1).detach()
            ligand_pos_traj[i+1] = x_lig_mu.detach() + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = F.softmax(he_lig_mu/temp, dim=-1).detach()

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]
        # # 3. get the final positions
        return {
            'pred': [ligand_node_traj[self.num_timesteps].log(), ligand_pos_traj[self.num_timesteps], ligand_halfedge_traj[self.num_timesteps].log()],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }
    
    def specify_guid(
    self, n_graphs, 
    protein_node, protein_pos, protein_batch, 
    off_target_node, off_target_pos, off_target_batch, 
    ligand_batch, halfedge_index, halfedge_batch, classifier,
    batch_lab=None, gui_strength=None, 
    bond_predictor=None, guidance=None, gen_mode=None
    ):
        B = protein_batch.max() + 1
        device = ligand_batch.device
        # # 1. get the init values (position, node and edge types)
        n_nodes_ligand = len(ligand_batch)
        n_halfedges_ligand = len(halfedge_batch)        
        node_init = self.node_transition.sample_init(n_nodes_ligand)
        halfedge_init = self.edge_transition.sample_init(n_halfedges_ligand)
        if self.categorical_space == 'discrete':
            _, ligand_node_h_init, log_node_type = node_init
            _, ligand_halfedge_h_init, log_halfedge_type = halfedge_init
        else:
            ligand_node_h_init = node_init
            ligand_halfedge_h_init = halfedge_init

        pocket_center_pos = scatter_mean(protein_pos, protein_batch, dim=0)
        ligand_center_pos = pocket_center_pos[ligand_batch]
        ligand_pos_init = ligand_center_pos + torch.randn_like(ligand_center_pos)
        protein_pos, ligand_pos_init, offset = center_pos(protein_pos, ligand_pos_init, protein_batch, ligand_batch, self.center_pos_mode)
        off_target_pos = off_target_pos - offset[off_target_batch]

        # # 1.1 init trajectory
        ligand_node_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, ligand_node_h_init.shape[-1]],
                                dtype=ligand_node_h_init.dtype).to(device)
        ligand_pos_traj = torch.zeros([self.num_timesteps + 1, n_nodes_ligand, 3], dtype=ligand_pos_init.dtype).to(device)
        ligand_halfedge_traj = torch.zeros([self.num_timesteps + 1, n_halfedges_ligand, ligand_halfedge_h_init.shape[-1]],
                                    dtype=ligand_halfedge_h_init.dtype).to(device)
        ligand_node_traj[0] = ligand_node_h_init
        ligand_pos_traj[0] = ligand_pos_init + offset[ligand_batch]
        ligand_halfedge_traj[0] = ligand_halfedge_h_init

        # # 2. sample loop
        ligand_node_h_pert = ligand_node_h_init
        ligand_pos_pert = ligand_pos_init
        ligand_halfedge_h_pert = ligand_halfedge_h_init
        ligand_edge_index = torch.cat([halfedge_index, halfedge_index.flip(0)], dim=1)
        ligand_edge_batch = torch.cat([halfedge_batch, halfedge_batch], dim=0)
        
        sigma_x0  = 0.0001

        x_lig_in  = ligand_pos_pert.detach().clone()
        c_lig_in  = ligand_node_h_pert.detach().clone()
        he_lig_in = ligand_halfedge_h_pert.detach().clone()

        x_lig_mu  = torch.autograd.Variable(x_lig_in.detach().clone(), requires_grad=True).float()
        c_lig_mu  = torch.autograd.Variable(c_lig_in.detach().clone(), requires_grad=True).float()
        he_lig_mu = torch.autograd.Variable(he_lig_in.detach().clone(), requires_grad=True).float()
        optimizer = Adam(params=[x_lig_mu, c_lig_mu, he_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
        temp = 0.5
        for i, step in tqdm(enumerate(range(self.num_timesteps)[::-1]), total=self.num_timesteps):
            time_step = torch.full(size=(n_graphs,), fill_value=step, dtype=torch.long).to(device)

            if i == 0:
                c_lig_mu_norm  = c_lig_mu
                he_lig_mu_norm = he_lig_mu
            else:
                c_lig_mu_norm  = F.softmax(c_lig_mu, dim=-1)
                he_lig_mu_norm = F.softmax(he_lig_mu, dim=-1)

            x0_pred  = x_lig_mu + sigma_x0 * torch.randn_like(x_lig_mu)
            c0_pred  = c_lig_mu_norm
            he0_pred = he_lig_mu_norm
            dist_c   = Categorical(probs=c0_pred)
            dist_he  = Categorical(probs=he0_pred)
            cv_pred  = dist_c.sample()
            hev_pred = dist_he.sample()

            x_lig_t  = self.pos_transition.add_noise(x0_pred, time_step, ligand_batch)
            c_lig_t, _, _  = self.node_transition.add_noise(cv_pred, time_step, ligand_batch)
            he_lig_t, _, _ = self.edge_transition.add_noise(hev_pred, time_step, halfedge_batch)
            e_lig_t  = torch.cat([he_lig_t, he_lig_t], dim=0)       
            
            # # 2.3 inference
            with torch.no_grad():
                preds = self(
                    protein_node, protein_pos, protein_batch,
                    c_lig_t, x_lig_t, ligand_batch,
                    e_lig_t, ligand_edge_index, ligand_edge_batch, 
                    time_step, batch_lab
                )
                pred_ligand_pos, pred_ligand_node, pred_ligand_halfedge = preds['pred_ligand_pos'], preds['pred_ligand_node'], preds['pred_ligand_halfedge']

                pred_x_0  = pred_ligand_pos
                pred_c_0  = F.softmax(pred_ligand_node, dim=-1)
                pred_he_0 = F.softmax(pred_ligand_halfedge, dim=-1)
            
            if i >= 600 and i <=900:
                prop_pred = classifier.predict(x_rec=protein_pos, x_lig=x_lig_mu, 
                                               h_rec=protein_node, h_lig=c_lig_mu_norm,
                                               batch_idx_rec=protein_batch, batch_idx_lig=ligand_batch)
                prop_pred = prop_pred.clamp(min=0.0, max=1.0)

                prop_pred_off = classifier.predict(x_rec=off_target_pos, x_lig=x_lig_mu, 
                                                   h_rec=off_target_node, h_lig=c_lig_mu_norm,
                                                   batch_idx_rec=off_target_batch, batch_idx_lig=ligand_batch)
                prop_pred_off = prop_pred_off.clamp(min=0.0, max=1.0)

                loss_prop = (((0.9 - prop_pred)**2)).mean()
                loss_prop_off = (((0.1 - prop_pred_off)**2)).mean()

                grad_prop_x = torch.autograd.grad(loss_prop, x_lig_mu, retain_graph=True)[0] * B
                grad_prop_c = torch.autograd.grad(loss_prop, c_lig_mu, retain_graph=True)[0] * B
                grad_propoff_x = torch.autograd.grad(loss_prop_off, x_lig_mu, retain_graph=True)[0] * B
                grad_propoff_c = torch.autograd.grad(loss_prop_off, c_lig_mu, retain_graph=False)[0] * B
                w = 0.1
            else:
                prop_pred = None
                prop_pred_off = None
                grad_prop_x = torch.zeros_like(x_lig_mu)
                grad_prop_c = torch.zeros_like(c_lig_mu_norm)
                grad_propoff_x = torch.zeros_like(x_lig_mu)
                grad_propoff_c = torch.zeros_like(c_lig_mu_norm)
                w = 0.0


            node_pt0 = self.node_transition.q_mats[step][None, ...].repeat(n_nodes_ligand, 1, 1)
            node_pT0 = self.node_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_nodes_ligand, 1, 1)
            he_pt0 = self.edge_transition.q_mats[step][None, ...].repeat(n_halfedges_ligand, 1, 1)
            he_pT0 = self.edge_transition.q_mats[self.num_timesteps-1][None, ...].repeat(n_halfedges_ligand, 1, 1)

            grad_x_noise = (x_lig_mu - pred_ligand_pos).detach()
            loss_noise_p0t, grad_c_p0t = self.loss_noise_p0t(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, batch_idx=ligand_batch)
            loss_noise_pt, grad_c_pt   = self.loss_noise_pt(c_lig_t, step, pt0=node_pt0, q0=c_lig_mu_norm, p0t=pred_c_0, pT0=node_pT0, batch_idx=ligand_batch)
            loss_noise_p0t, grad_he_p0t = self.loss_noise_p0t(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, batch_idx=halfedge_batch)
            loss_noise_pt, grad_he_pt   = self.loss_noise_pt(he_lig_t, step, pt0=he_pt0, q0=he_lig_mu_norm, p0t=pred_he_0, pT0=he_pT0, batch_idx=halfedge_batch)

            grad_c_noise = grad_c_p0t + grad_c_pt
            grad_he_noise = grad_he_p0t + grad_he_pt
            grad_c_norm = grad_c_noise
            grad_he_norm = grad_he_noise

            if torch.isnan(grad_c_noise).any().item() == True or torch.isinf(grad_c_noise).any().item() == True:
                breakpoint()
            if torch.isnan(grad_he_noise).any().item() == True or torch.isinf(grad_he_noise).any().item() == True:
                breakpoint()

            grad_x = 0.25 * grad_x_noise + w * grad_prop_x + w * grad_propoff_x
            jacob = self.softmax_jacobian(c_lig_mu, temp)
            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]
            grad_c = grad_c + w * grad_prop_c + w * grad_propoff_c
            jacob = self.softmax_jacobian(he_lig_mu, temp)
            grad_he = torch.matmul(jacob.transpose(-1, -2), grad_he_norm[:, :, None])[:, :, 0]

            optimizer.zero_grad()
            x_lig_mu.grad  = grad_x
            c_lig_mu.grad  = grad_c
            he_lig_mu.grad = grad_he
            optimizer.step()

            # 2.6 update trajectory
            ligand_node_traj[i+1] = F.softmax(c_lig_mu/temp, dim=-1).detach()
            ligand_pos_traj[i+1] = x_lig_mu.detach() + offset[ligand_batch]
            ligand_halfedge_traj[i+1] = F.softmax(he_lig_mu/temp, dim=-1).detach()

        pred_ligand_pos = pred_ligand_pos + offset[ligand_batch]

        # # 3. get the final positions
        return {
            'pred': [ligand_node_traj[self.num_timesteps].log(), ligand_pos_traj[self.num_timesteps], ligand_halfedge_traj[self.num_timesteps].log()],
            'traj': [ligand_node_traj, ligand_pos_traj, ligand_halfedge_traj]
        }

    def bond_guidance(self, gui_type, gui_scale, pred_bondpredictor, ligand_pos_in, halfedge_type_prev, log_halfedge_type):
        if gui_type == 'entropy':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'uncertainty':
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log().sum()
            # t2 = time.time()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
            # t3 = time.time()
        elif gui_type == 'uncertainty_bond':  # only for the predicted real bond (not no bond)
            prob = torch.softmax(pred_bondpredictor, dim=-1)
            uncertainty = torch.sigmoid( -torch.logsumexp(pred_bondpredictor, dim=-1))
            uncertainty = uncertainty.log()
            uncertainty = (uncertainty * prob[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(uncertainty, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'entropy_bond':
            prob_halfedge = torch.softmax(pred_bondpredictor, dim=-1)
            entropy = - torch.sum(prob_halfedge * torch.log(prob_halfedge + 1e-12), dim=-1)
            entropy = entropy.log()
            entropy = (entropy * prob_halfedge[:, 1:].detach().sum(dim=-1)).sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit_bond':
            ind_real_bond = ((halfedge_type_prev >= 1) & (halfedge_type_prev <= 4))
            idx_real_bond = ind_real_bond.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'logit':
            ind_bond_notmask = (halfedge_type_prev <= 4)
            idx_real_bond = ind_bond_notmask.nonzero().squeeze(-1)
            pred_real_bond = pred_bondpredictor[idx_real_bond, halfedge_type_prev[idx_real_bond]]
            pred = pred_real_bond.sum()
            delta = + torch.autograd.grad(pred, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent':
            prob_halfedge_type = log_halfedge_type.exp()[:, :-1]  # the last one is masked bond (not used in predictor)
            entropy = F.cross_entropy(pred_bondpredictor, prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        elif gui_type == 'crossent_bond':
            prob_halfedge_type = log_halfedge_type.exp()[:, 1:-1]  # the last one is masked bond. first one is no bond
            entropy = F.cross_entropy(pred_bondpredictor[:, 1:], prob_halfedge_type, reduction='none')
            entropy = entropy.log().sum()
            delta = - torch.autograd.grad(entropy, ligand_pos_in)[0] * gui_scale
        else:
            raise NotImplementedError(f'Guidance type {gui_type} is not implemented')
        
        return delta

