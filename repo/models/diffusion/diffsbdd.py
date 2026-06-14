import torch
from torch.optim import Adam
from .diffusion_scheduler import VariationalScheduler, DiffsbddVariationalScheduler
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from .._base import register_model
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *
# from repo.models.utils.savemol import reconstruct_save_mol
import torch.nn.functional as F
from repo.modules.common import compose_context, get_dict_mean
from repo.models.utils.u import kabsch_mse, kabsch_var_mse, kabsch_var_mse_scale, optimize_R, optimize_SE3
from tqdm.auto import tqdm
from ._base import BaseDiff
from torch_scatter import scatter_mean, scatter_add
import math
from ..utils.categorical import index_to_log_onehot
import time
import copy
from repo.models.charge_density.utils import prepare_input, prepare_input_given_grid


def zero_com_translate(x_lig_pred, batch_idx_lig, x_composed, lig_flag_composed):
    noise_lig_pred = x_lig_pred - x_composed[lig_flag_composed]
    noise_lig_pred_mean = scatter_mean(noise_lig_pred, batch_idx_lig, dim=0)[batch_idx_lig]
    noise_lig_pred = noise_lig_pred - noise_lig_pred_mean
    return noise_lig_pred


@register_model('diffsbdd')
class DiffSBDD(BaseDiff):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.num_classes = cfg.num_atomtype
        
        self.pos_scheduler = DiffsbddVariationalScheduler(self.num_diffusion_timesteps, 
                                                  type = pos_scheduler_cfg.type)
        
        
        atom_scheduler_cfg = cfg.generator.atom_schedule
        self.type_scheduler = DiffsbddVariationalScheduler(self.num_diffusion_timesteps,
                                                   type = atom_scheduler_cfg.type) 
        cfg.embedder.num_atomtype = cfg.num_atomtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)

        self.intersect_reg = cfg.get('intersect_reg', True)

    def forward(self, batch): 
        x_lig_0 = batch['ligand_pos']
        v_lig_0 = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        N_lig, _ = x_lig_0.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        if self.training:
            t = self.sample_time(B, device = x_lig_0.device, ctn=True)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                 lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                                 gen_flag_lig, gen_flag_rec, t)
        
        else:
            loss_dicts = []
            results = []
            eval_times = np.linspace(1, 
                                     self.num_diffusion_timesteps, 
                                     self.cfg.get('eval_interval', 10))
            for t in eval_times:
                t = torch.tensor([t] * B).long().to(x_lig_0.device)
                loss_dict, result = self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                                  lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                                                  gen_flag_lig, gen_flag_rec, t)
                loss_dicts.append(loss_dict)
                results.append(result)
            
            loss_dict_mean = get_dict_mean(loss_dicts)

            return loss_dict_mean, results

    def normalize_pos(self, pos, std=1, mean=0):
        return (pos - mean) / std

    def normalize_type(self, c, std=4, mean=0):
        return (c - mean) / std

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                  lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                  gen_flag_lig, gen_flag_rec, t):
        

        x_lig_0 = self.normalize_pos(x_lig_0)
        x_rec_0 = self.normalize_pos(x_rec_0)

        c_lig_0 = F.one_hot(v_lig_0, self.num_classes)
        c_lig_0 = self.normalize_type(c_lig_0)
        v_rec_0 = self.normalize_type(v_rec_0)

        s_int = t - 1
        t_is_zero = (t == 0).float()
        t_is_not_zero = 1 - t_is_zero
        s = s_int / self.num_diffusion_timesteps
        t = t / self.num_diffusion_timesteps

        if self.denoise_structure:
            x_lig_0, x_rec_0 = self.pos_scheduler.remove_mean_batch(x_lig_0, x_rec_0, batch_idx_lig, batch_idx_rec)
            x_lig_t, pos_noise, x_rec_t = self.pos_scheduler.forward_pos_center_noise((x_lig_0, x_rec_0), t, (batch_idx_lig, batch_idx_rec), gen_flag_lig, zero_center=False)

        if self.denoise_atom:
            c_lig_t, type_noise = self.type_scheduler.forward_type_add_noise(c_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = c_lig_0

        x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]

        x_lig_pred_non_training = None
        pos_noise_non_training = None
        c_lig_pred_non_training = None
        type_noise_non_training = None

        if not self.training:
            t_zeros = torch.zeros_like(s)
            if self.denoise_structure:
                x_lig_t, pos_noise_non_training, x_rec_t = self.pos_scheduler.forward_pos_center_noise((x_lig_0, x_rec_0), t_zeros, (batch_idx_lig, batch_idx_rec), gen_flag_lig, zero_center=False)

            if self.denoise_atom:
                c_lig_t, type_noise_non_training = self.type_scheduler.forward_type_add_noise(c_lig_0, t_zeros, batch_idx_lig, gen_flag_lig)
            else:
                c_lig_t = c_lig_0
            
            x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                    batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_zeros)
            
            context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                            {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                            batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
            x_lig_pred_non_training = x[context_composed['lig_flag']]
            c_lig_pred_non_training = v[context_composed['lig_flag']]

        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_score_loss(x_lig_pred, pos_noise, t, 
                                                                   gen_flag_lig, batch_idx_lig, 
                                                                   score_in=False, info_tag='pos', 
                                                                   s=s, x_lig_0=x_lig_0, 
                                                                   compute_continus=True,
                                                                   t_is_zero=t_is_zero,
                                                                   t_is_not_zero=t_is_not_zero,
                                                                   x_pred_t_non_training=x_lig_pred_non_training,
                                                                   x_tgt_t_non_training=pos_noise_non_training,
                                                                   )
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}

        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_score_loss(c_lig_pred, type_noise, t, 
                                                                      gen_flag_lig, batch_idx_lig, 
                                                                      score_in=False, info_tag='atom',
                                                                      s=s, c_lig_0=c_lig_0, 
                                                                      c_lig_t=c_lig_t, compute_discrete=True,
                                                                      t_is_zero=t_is_zero,
                                                                      t_is_not_zero=t_is_not_zero,      
                                                                      c_pred_t_non_training=c_lig_pred_non_training,
                                                                      c_tgt_t_non_training=type_noise_non_training,
                                                                      )
        else:
            loss_atom, atom_info = torch.tensor(0).float(), {}

        results = {}
        results.update(pos_info)
        results.update(atom_info)

        return {'pos': loss_pos, 'atom': loss_atom}, results
    
    def unnormalize_pos(self, x, std=1, mean=0):
        return x * std + mean
    
    def unnormalize_type(self, c, std=4, mean=0):
        return c * std + mean

    def zero_time_loss(self, c_lig_0, c_lig_t, pos_noise, x_lig_pred, t, gen_flag_lig, batch_idx_lig, epsilon=1e-7): 
        gamma = self.type_scheduler.gamma(t)[batch_idx_lig]
        sigma_0 = self.unnormalize_type(torch.sqrt(torch.sigmoid(gamma))) 

        log_p_x_given_z0_without_constants_ligand = -0.5 * (
            (pos_noise - x_lig_pred) ** 2
        )

        ligand_onehot = self.unnormalize_type(c_lig_0)
        estimated_ligand_onehot = self.unnormalize_type(c_lig_t)
        centered_ligand_onehot = estimated_ligand_onehot - 1

        log_ph_cat_proportional_ligand = torch.log(
            self.cdf_standard_gaussian((centered_ligand_onehot + 0.5) / sigma_0.unsqueeze(-1))
            - self.cdf_standard_gaussian((centered_ligand_onehot - 0.5) / sigma_0.unsqueeze(-1))
            + epsilon
        )

        log_Z = torch.logsumexp(log_ph_cat_proportional_ligand, dim=1, keepdim=True)
        log_probabilities_ligand = log_ph_cat_proportional_ligand - log_Z

        log_ph_given_z0_ligand = log_probabilities_ligand * ligand_onehot

        t_zero_mask = (t == 0)[batch_idx_lig]
        gen_mask = torch.logical_and(gen_flag_lig, t_zero_mask)

        if gen_mask.sum() > 0:
            loss_pos = - scatter_mean(log_p_x_given_z0_without_constants_ligand.sum(-1)[gen_mask],
                                      batch_idx_lig[gen_mask], dim=0)
            loss_atom = - scatter_mean(log_ph_given_z0_ligand.sum(-1)[gen_mask], 
                                       batch_idx_lig[gen_mask], dim=0)
        else:
            loss_pos = torch.zeros_like(log_p_x_given_z0_without_constants_ligand.sum(-1))
            loss_atom = torch.zeros_like(log_ph_given_z0_ligand.sum(-1))
        
        return loss_pos.mean(0), loss_atom.mean(0)

    @staticmethod
    def cdf_standard_gaussian(x):
        return 0.5 * (1. + torch.erf(x / math.sqrt(2)))

    def frag_pocket_piror(self, mu_lig, xh0_pocket, x0_lig, gen_mask, lig_idx,
                               pocket_idx, com=False):
        
        mu_lig_norm = torch.linalg.norm(mu_lig, dim=-1, keepdim=True)
        mu_lig = (1. + 1./(mu_lig_norm+1.)) * mu_lig

        B = lig_idx.max().item()+1
        cov = torch.cov(x0_lig[lig_idx==0][~gen_mask[lig_idx==0]].T)
        cov = cov / torch.det(cov)**(1/3)
        cov = cov[None, ...].repeat(mu_lig.size(0), 1, 1) # frag are same alone batch
        # cov = torch.zeros(B, mu_lig.size(1), mu_lig.size(1), device=lig_idx.device)
        # for i in range(B):
        #     cov[i] = torch.cov(x0_lig[lig_idx==i][~gen_mask[lig_idx==i]].T)
        # cov = cov[lig_idx]
        from torch.distributions import MultivariateNormal
        dist = MultivariateNormal(mu_lig, cov)
        out_lig = dist.sample()
        # project to COM-free subspace
        if com:
            xh_pocket = xh0_pocket.detach().clone()
            out_lig, xh_pocket = \
                self.pos_scheduler.remove_mean_batch(out_lig,
                                    xh0_pocket,
                                    lig_idx, pocket_idx)
            return out_lig, xh_pocket
        else:
            return out_lig
    
    def sample(self, batch):
        torch.set_grad_enabled(False)
        self.eval()

        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()

        x_rec_0 = self.normalize_pos(x_rec_0)
        v_rec_0 = self.normalize_type(v_rec_0)

        n_samples = batch_idx_lig.max() + 1
        mu_lig_X = scatter_mean(x_rec_0, batch_idx_rec, dim=0)[batch_idx_lig]
        mu_lig_h = torch.zeros((n_samples, self.num_classes), device=x_rec_0.device)[batch_idx_lig]
        sigma = torch.ones_like(torch.bincount(batch_idx_rec)).unsqueeze(1)

        x_lig_in, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(mu_lig_X, x_rec_0, sigma, batch_idx_lig, batch_idx_rec, com=True)

        v_lig_in = self.pos_scheduler.sample_normal_zero_com(mu_lig_h, v_rec_0, sigma, batch_idx_lig, batch_idx_rec)
        
        self.pos_scheduler.assert_mean_zero_with_mask(x_lig_in, batch_idx_lig)

        c_lig_in = v_lig_in

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))

        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]
            s_array = torch.full((n_samples,), fill_value=t_idx,
                                 device=x_lig.device)
            t_array = s_array + 1
            s_array = s_array / self.num_diffusion_timesteps
            t_array = t_array / self.num_diffusion_timesteps

            x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_array)
        
            context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                             {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            x_lig_pred = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]

            if self.denoise_structure:
                x_lig_next, x_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                            s_array, t_array, x_lig, x_rec_0, batch_idx_lig, batch_idx_rec, x_lig_pred, com=True)
            else:
                x_lig_next = x_lig

            if self.denoise_atom:
                c_lig_next, v_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                            s_array, t_array, c_lig, v_rec_0, batch_idx_lig, batch_idx_rec, c_lig_out, com=False)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (x_lig_next.clone(), c_lig_next.clone(), batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 


        x_lig, c_lig, _ = self.sample_p_xh_given_z0(x_lig_next, c_lig_next, x_rec_0, v_rec_0, aa_rec_0, 
                                                 batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                                                 gen_flag_lig, gen_flag_rec)


        traj[0] = (x_lig.cpu(), c_lig.cpu(), batch_idx_lig.cpu())    
        return traj
      
    def inpaint(self, batch, resamples=10):

        x_rec_0 = batch['protein_pos']
        x_lig_0 = batch['ligand_pos']
        v_rec_0 = batch['protein_atom_feature']
        v_lig_0 = batch['ligand_atom_type']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()

        if gen_flag_lig.all().item():
            return []
        
        x_rec_0 = self.normalize_pos(x_rec_0)
        v_rec_0 = self.normalize_type(v_rec_0)
        x_lig_0 = self.normalize_pos(x_lig_0)
        v_lig_0 = self.normalize_type(v_lig_0)

        com_rec_0 = scatter_mean(x_rec_0, batch_idx_rec, dim=0)

        n_samples = batch_idx_lig.max() + 1
        mu_lig_x = scatter_mean(x_lig_0[gen_flag_lig], batch_idx_lig[gen_flag_lig], dim=0)[batch_idx_lig]
        mu_lig_h = torch.zeros((n_samples, self.num_classes), device=x_rec_0.device)[batch_idx_lig]
        sigma = torch.ones_like(torch.bincount(batch_idx_rec)).unsqueeze(1)
        
        x_lig_in, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(
            mu_lig_x, x_rec_0, sigma, batch_idx_lig, batch_idx_rec, com=True)
        
        v_lig_in = self.pos_scheduler.sample_normal_zero_com(mu_lig_h, v_rec_0, sigma, batch_idx_lig, batch_idx_rec)

        self.pos_scheduler.assert_mean_zero_with_mask(x_lig_in, batch_idx_lig)

        c_lig_in = v_lig_in

        time_steps = int(2 * self.num_diffusion_timesteps / resamples)
        true_t = lambda t: (t / time_steps * self.num_diffusion_timesteps)
        time_seq = list(reversed(range(0, time_steps)))

        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1
        diff_mask = torch.ones_like(gen_flag_lig)

        traj = {time_steps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]
            s_array = torch.full((n_samples,), fill_value=t_idx,
                                 device=x_lig.device)
            t_array = s_array + 1
            s_array = s_array / time_steps
            t_array = t_array / time_steps

            for r in range(resamples):
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_array)

                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': diff_mask, 'lig_flag':lig_flag},
                                                                 {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_pred = x[context_composed['lig_flag']]
                c_lig_out = v[context_composed['lig_flag']]

                if self.denoise_structure:
                    # sample inpainted part
                    x_lig_next, x_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                         s_array, t_array, x_lig, x_rec_0, batch_idx_lig, batch_idx_rec,
                         x_lig_pred, com=True)
                else:
                    x_lig_next = x_lig

                if self.denoise_atom:
                    c_lig_next, v_rec_0 = self.pos_scheduler.sample_p_zs_given_zt(
                            s_array, t_array, c_lig, v_rec_0, batch_idx_lig, batch_idx_rec, 
                            c_lig_out, com=False)
                else:
                    c_lig_next = c_lig

                if r < resamples - 1:
                    if self.denoise_structure:
                        if torch.isnan(x_lig_next).any().item():
                            breakpoint()
                        x_lig_unknown, x_rec_0 = self.pos_scheduler.sample_p_zt_given_zs(
                            x_lig_next, x_rec_0, batch_idx_lig, batch_idx_rec, s_array, 
                            t_array, com=True)
                        com_rec = scatter_mean(x_rec_0, batch_idx_rec, dim=0)
                        x_lig_mu_align = x_lig_0 + (com_rec - com_rec_0)[batch_idx_lig]
                        x_lig_known, pos_noise, x_rec_0 = self.pos_scheduler.forward_pos_center_noise(
                            (x_lig_mu_align, x_rec_0), t_array, (batch_idx_lig, batch_idx_rec), 
                            diff_mask, zero_center=False, com=True)
                        com_noised = scatter_mean(x_lig_known[~gen_flag_lig.bool().view(-1)],
                                              batch_idx_lig[~gen_flag_lig.bool().view(-1)], dim=0)
                        com_denoised = scatter_mean(x_lig_unknown[~gen_flag_lig.bool().view(-1)],
                                                    batch_idx_lig[~gen_flag_lig.bool().view(-1)], dim=0)
                        x_lig_known = x_lig_known + (com_denoised - com_noised)[batch_idx_lig]
                        x_rec_0 = x_rec_0 + (com_denoised - com_noised)[batch_idx_rec]
                        x_lig = x_lig_unknown * gen_flag_lig[:, None].float() + \
                                (1 - gen_flag_lig[:, None].float()) * x_lig_known
                    else:
                        x_lig = x_lig_next

                    if self.denoise_atom:
                        c_lig_unknown, v_rec_0 = self.pos_scheduler.sample_p_zt_given_zs(
                            c_lig_next, v_rec_0, batch_idx_lig, batch_idx_rec, s_array, 
                            t_array, com=False)
                        c_lig_known, type_noise = self.pos_scheduler.forward_type_add_noise(
                                v_lig_0, t_array, batch_idx_lig, diff_mask)
                        c_lig = c_lig_unknown * gen_flag_lig[:, None].float() + \
                                (1 - gen_flag_lig[:, None].float()) * c_lig_known
                    else:
                        c_lig = c_lig_next
                    
                    if c_lig.max().item() > 100:  # gen failed
                        return []
            
            traj[t_idx - 1] = (x_lig_next.clone(), c_lig_next.clone(), batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 


        x_lig, c_lig, x_rec_0 = self.sample_p_xh_given_z0(x_lig_next, c_lig_next, x_rec_0, v_rec_0, aa_rec_0, 
                                                 batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                                                 gen_flag_lig, gen_flag_rec)

        traj[0] = (x_lig.cpu(), c_lig.cpu(), batch_idx_lig.cpu())    
        return traj
    
    def dr_slover(self, batch, ts, show=True):

        x_rec_0 = batch['protein_pos']
        x_lig_0 = batch['ligand_pos']
        v_rec_0 = batch['protein_atom_feature']
        v_lig_0 = batch['ligand_atom_type']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()

        if gen_flag_lig.all().item():
            return []

        x_rec_0 = self.normalize_pos(x_rec_0)
        v_rec_0 = self.normalize_type(v_rec_0)
        x_lig_0 = self.normalize_pos(x_lig_0)
        v_lig_0 = self.normalize_type(v_lig_0)

        com_rec_0 = scatter_mean(x_rec_0, batch_idx_rec, dim=0)

        n_samples = batch_idx_lig.max() + 1
        mu_lig_x = scatter_mean(x_rec_0, batch_idx_rec, dim=0)[batch_idx_lig]
        mu_lig_h = torch.zeros((n_samples, self.num_classes), device=x_rec_0.device)[batch_idx_lig]
        sigma = torch.ones_like(torch.bincount(batch_idx_rec)).unsqueeze(1)

        x_lig_unknown, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(
            mu_lig_x, x_rec_0, sigma, batch_idx_lig, batch_idx_rec, com=True)
            
        com_rec = scatter_mean(x_rec_0, batch_idx_rec, dim=0)
        x_lig_0 = x_lig_0 + (com_rec - com_rec_0)[batch_idx_lig]
        x_lig_known = x_lig_0

        com_noised = scatter_mean(x_lig_known[~gen_flag_lig.bool().view(-1)],
                                    batch_idx_lig[~gen_flag_lig.bool().view(-1)], dim=0)
        com_denoised = scatter_mean(x_lig_unknown[~gen_flag_lig.bool().view(-1)],
                                    batch_idx_lig[~gen_flag_lig.bool().view(-1)], dim=0)
        x_lig_known = x_lig_known + (com_denoised - com_noised)[batch_idx_lig]
        x_rec_0 = x_rec_0 + (com_denoised - com_noised)[batch_idx_rec]
        x_lig_0 = x_lig_0 + (com_denoised - com_noised)[batch_idx_lig]

        v_lig_unknown = self.pos_scheduler.sample_normal_zero_com(
            mu_lig_h, v_rec_0, sigma, batch_idx_lig, batch_idx_rec)
        v_lig_known = v_lig_0.detach().clone()

        x_lig_in = x_lig_unknown * gen_flag_lig[:, None].float() + \
                (1 - gen_flag_lig[:, None].float()) * x_lig_known
        v_lig_in = v_lig_unknown * gen_flag_lig[:, None].float() + \
                (1 - gen_flag_lig[:, None].float()) * v_lig_known

        self.pos_scheduler.assert_mean_zero_with_mask(x_lig_in, batch_idx_lig)

        c_lig_in = v_lig_in

        x_lig_mu = torch.autograd.Variable(x_lig_in, requires_grad=True)
        c_lig_mu = torch.autograd.Variable(c_lig_in, requires_grad=True)
        optimize_mask = torch.ones_like(gen_flag_lig, dtype=torch.bool)
        h_mask = 1. - (gen_flag_lig).float()[:, None]

        optimizer = Adam([x_lig_mu, c_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
        traj = {ts[-1]: (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())}
        for t_idx in tqdm(ts, desc='sampling', total=len(ts)) if show else ts:
            
            t_array = torch.full((n_samples,), fill_value=t_idx, 
                                 device=x_lig_mu.device)
            s_array = t_array - 1
            t_array = t_array / self.num_diffusion_timesteps
            s_array = s_array / self.num_diffusion_timesteps

            sigma_x0 = 0.0001
            sigma_c0 = 0.0001
            noise_x0 = torch.randn_like(x_lig_mu)
            noise_c0 = torch.randn_like(c_lig_mu)
            
            x0_pred = x_lig_mu + sigma_x0 * noise_x0
            c0_pred = c_lig_mu + sigma_c0 * noise_c0
            
            x_lig_t, pos_noise, x_rec_t = self.pos_scheduler.forward_pos_center_noise(
                (x0_pred, x_rec_0), t_array, (batch_idx_lig, batch_idx_rec), 
                optimize_mask, zero_center=False)
            c_lig_t, type_noise = self.type_scheduler.forward_type_add_noise(
                c0_pred, t_array, batch_idx_lig, optimize_mask)

            with torch.no_grad():
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                   batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_array)
                
                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': optimize_mask, 'lig_flag':lig_flag},
                                                                 {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_pred = x[context_composed['lig_flag'].bool()]
                c_lig_pred = v[context_composed['lig_flag'].bool()]

            snr = self.pos_scheduler.SNR(self.pos_scheduler.gamma(t_array))
            
            loss_x_obs = (h_mask * (x_lig_0 - x_lig_mu)**2).sum() / h_mask.sum()
            loss_x_noise = ((x_lig_pred - pos_noise).detach() * x0_pred).mean()
            loss_c_obs = (h_mask * (v_lig_0 - c_lig_mu)**2).sum() / h_mask.sum()
            loss_c_noise = ((c_lig_pred - type_noise).detach() * c0_pred).mean()
            loss = (loss_x_obs + 0.25/snr[0]*loss_x_noise) + (loss_c_obs + 0.25/snr[0]*loss_c_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            traj[t_idx - 1] = (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())

        return traj
    
    def charge_density_guid(self, batch, ts, classifier, params, local=False):

        x_rec_0 = batch['protein_pos']
        x_lig_0 = batch['ligand_pos']
        v_rec_0 = batch['protein_atom_feature']
        v_lig_0 = batch['ligand_atom_type']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))
        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()

        x_rec_0 = self.normalize_pos(x_rec_0)
        v_rec_0 = self.normalize_type(v_rec_0)
        x_lig_0 = self.normalize_pos(x_lig_0)
        v_lig_0 = self.normalize_type(v_lig_0)

        B = batch_idx_lig.max() + 1

        tar_prop = params['tar_density'][None, :].repeat(B, 1)
        gird_pos = params['grid_pos'] - batch.protein_translation[:1].cpu().numpy()
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

        n_samples = batch_idx_lig.max() + 1
        mu_lig_x = scatter_mean(x_rec_0, batch_idx_rec, dim=0)[batch_idx_lig]
        mu_lig_h = torch.zeros((n_samples, self.num_classes), device=x_rec_0.device)[batch_idx_lig]
        sigma = torch.ones_like(torch.bincount(batch_idx_rec)).unsqueeze(1)

        x_lig_in, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(
                mu_lig_x, x_rec_0, sigma, batch_idx_lig, batch_idx_rec, com=True)
        v_lig_in = self.pos_scheduler.sample_normal_zero_com(
                mu_lig_h, v_rec_0, sigma, batch_idx_lig, batch_idx_rec)

        self.pos_scheduler.assert_mean_zero_with_mask(x_lig_in, batch_idx_lig)

        c_lig_in = v_lig_in

        x_lig_mu = torch.autograd.Variable(x_lig_in, requires_grad=True)
        c_lig_mu = torch.autograd.Variable(c_lig_in, requires_grad=True)
        optimize_mask = torch.ones_like(gen_flag_lig, dtype=torch.bool)

        loss_list = [[], [], []]
        optimizer = Adam([x_lig_mu, c_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)
        traj = {ts[-1]: (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())}
        if local:
            t_add = 800
            ts_seq = ts + list(range(-1, -t_add, -1))
        else:
            t_add = 0
            ts_seq = ts
        for t_idx in tqdm(ts_seq, desc='sampling', total=len(ts_seq)):
            
            add_t_idx = 0 if t_idx >=0 else t_idx
            if t_idx < 0:
                t_idx = np.random.randint(0, 50)
            t_idx = max(0, t_idx)
            t_idx = t_idx % self.num_diffusion_timesteps
            t_array = torch.full((n_samples,), fill_value=t_idx, 
                                 device=x_lig_mu.device)
            s_array = t_array - 1
            t_array = t_array / self.num_diffusion_timesteps
            s_array = s_array / self.num_diffusion_timesteps

            sigma_x0 = 0.0001
            sigma_c0 = 0.0001
            noise_x0 = torch.randn_like(x_lig_mu)
            noise_c0 = torch.randn_like(c_lig_mu)
            
            x0_pred = x_lig_mu + sigma_x0 * noise_x0
            c0_pred = c_lig_mu + sigma_c0 * noise_c0
            
            x_lig_t, pos_noise, x_rec_t = self.pos_scheduler.forward_pos_center_noise(
                (x0_pred, x_rec_0), t_array, (batch_idx_lig, batch_idx_rec), 
                optimize_mask, zero_center=False)
            c_lig_t, type_noise = self.type_scheduler.forward_type_add_noise(
                c0_pred, t_array, batch_idx_lig, optimize_mask)

            with torch.no_grad():
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_t, c_lig_t, v_rec_0, aa_rec_0, 
                                                                   batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_array)
                
                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': optimize_mask, 'lig_flag':lig_flag},
                                                                 {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_pred = x[context_composed['lig_flag']]
                c_lig_pred = v[context_composed['lig_flag']]

            # optimize more steps
            if local:
                allow_t = -1
                use_flag = add_t_idx < -400
            else:
                allow_t = 400
                use_flag = t_idx < allow_t
            if use_flag:
                c_net_in = self.unnormalize_type(c_lig_mu)
                c_net_in = F.softmax(c_net_in/0.1, dim=-1)
                
                input_dict = copy.deepcopy(params)
                input_dict['grid_pos'] = gird_pos
                cls_batch = prepare_input_given_grid((x_lig_mu.detach().cpu(), 
                                           c_net_in.detach().cpu(), 
                                           batch_idx_lig.detach().cpu()),
                                           device=x_lig_mu.device, 
                                           params=input_dict,
                                           label=tar_prop.cpu().numpy(),
                                           local=local)
                cls_batch['atom_xyz'].requires_grad = True
                prop_pred = classifier(cls_batch)
                prop_pred = prop_pred.clamp(min=0.0)
                tar_rho = cls_batch['label']

                if local is False:
                    tar_rho = tar_rho / torch.norm(tar_rho, dim=-1, keepdim=True)
                    prop_pred = prop_pred / torch.norm(prop_pred, dim=-1, keepdim=True)
                    high_rho = tar_rho.detach().clone()
                    peak_rho = tar_rho.detach().clone()
                    high_rho[high_rho < 0.010] = -0.02   # 0.01
                    high_rho[high_rho >= 0.010] = 0.0
                    peak_rho[peak_rho < 0.05] = 0.0
                    peak_rho[peak_rho >= 0.05] = 0.02
                    loss_high = - (high_rho * prop_pred).sum(-1).mean(0)
                    loss_peak = - (peak_rho * prop_pred).sum(-1).mean(0)
                    loss_prop = loss_high + loss_peak

                    w = 0.0 * t_idx / allow_t + 20.0 * t_idx / allow_t

                    loss_list[0].append(loss_prop.item())
                    loss_list[1].append(loss_high.item())
                    loss_list[2].append(loss_peak.item())
                else:
                    scale = tar_rho.sum(-1, keepdim=True) / prop_pred.sum(-1, keepdim=True)
                    prop_pred = scale * prop_pred
                    tar_rho = tar_rho.clamp(min=1.e-3, max=10)
                    prop_pred = prop_pred.clamp(min=1.e-3, max=10)
                    loss_shape = (prop_pred - tar_rho).abs().sum(-1) / tar_rho.abs().sum(-1)
                    loss_shape = loss_shape.mean()
                    loss_scale = (scale - 1.).abs().mean()
                    loss_prop = loss_shape + 0. * loss_scale

                    w = 25.0
                    
                    loss_list[0].append(loss_prop.item())
                    loss_list[1].append(loss_shape.item())
                    loss_list[2].append(loss_scale.item())

                grad_prop_x = torch.autograd.grad(loss_prop, cls_batch['atom_xyz'], retain_graph=False)[0] * B
                grad_prop_x = grad_prop_x.reshape(-1, 3)[cls_batch['heavy_mask'].reshape(-1)]
            
                grad_prop_c = torch.zeros_like(c_lig_mu)
                if torch.isnan(grad_prop_x).any().item() == True:
                    mask_nan = torch.isnan(grad_prop_x)
                    grad_prop_x[mask_nan] = 0.
            else:
                prop_pred = None
                grad_prop_x = torch.zeros_like(x_lig_mu)
                grad_prop_c = torch.zeros_like(c_lig_mu)
                loss_prop = torch.zeros(1, device=grad_prop_c.device).mean()
                w = 0.0

            alpha_t = self.pos_scheduler.alpha(self.pos_scheduler.gamma(t_array), x_lig_t)
            sigma_t = self.pos_scheduler.sigma(self.pos_scheduler.gamma(t_array), x_lig_t)
            
            alpha_t = self.pos_scheduler.alpha(self.pos_scheduler.gamma(t_array), x_lig_t)
            sigma_t = self.pos_scheduler.sigma(self.pos_scheduler.gamma(t_array), x_lig_t)
            x_mu_hat = (x_lig_t - sigma_t[batch_idx_lig]*x_lig_pred) / alpha_t[batch_idx_lig]
            c_mu_hat = (c_lig_t - sigma_t[batch_idx_lig]*c_lig_pred) / alpha_t[batch_idx_lig]
            
            grad_noise_x = (x_lig_mu - x_mu_hat).detach()
            grad_noise_c = (c_lig_mu - c_mu_hat).detach()
            
            grad_x = grad_noise_x + w * grad_prop_x
            grad_c = grad_noise_c + w * grad_prop_c

            optimizer.zero_grad()
            x_lig_mu.grad = grad_x
            c_lig_mu.grad = grad_c
            optimizer.step()
            
            atom_dist = torch.topk(torch.norm(x_mu_hat[None, ...]-x_mu_hat[:, None, :], dim=-1), 
                                   k=2, dim=0, largest=False)[0][1]

            traj[t_idx] = (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())
        print(f'{loss_list[0][-1]}\t{loss_list[1][-1]}\t{loss_list[2][-1]}')
        return traj

    def sample_p_xh_given_z0(self, x_lig, c_lig, x_rec_0, v_rec_0, aa_rec_0, 
                             batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, 
                             gen_flag_lig, gen_flag_rec):
        B = batch_idx_lig.max() + 1
        t_zeros = torch.zeros(size=(B, )).to(x_lig)
        gamma_0 = self.pos_scheduler.gamma(t_zeros)
        sigma_0 = torch.exp(0.5 * gamma_0).unsqueeze(1)
        

        x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                           batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_zeros)

        context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                         {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                         batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_out = v[context_composed['lig_flag']]

        mu_x_lig = self.compute_pred(x_lig_pred, x_lig, gamma_0, batch_idx_lig)
        mu_c_lig = self.compute_pred(c_lig_out, c_lig, gamma_0, batch_idx_lig)

        x_lig_in, x_rec_0 = self.pos_scheduler.sample_normal_zero_com(mu_x_lig, x_rec_0, sigma_0, batch_idx_lig, batch_idx_rec, com=True)

        v_lig_in = self.pos_scheduler.sample_normal_zero_com(mu_c_lig, v_rec_0, sigma_0, batch_idx_lig, batch_idx_rec)        

        x_lig = self.unnormalize_pos(x_lig_in)
        x_rec_0 = self.unnormalize_pos(x_rec_0)
        c_lig = self.unnormalize_type(c_lig)
        
        return x_lig, c_lig, x_rec_0


    def compute_pred(self, net_out_lig, zt, gamma_t, batch_idx_lig):
        """Commputes x_pred, i.e. the most likely prediction of x."""
        sigma_t = self.pos_scheduler.sigma(gamma_t, target_tensor=net_out_lig)
        alpha_t = self.pos_scheduler.alpha(gamma_t, target_tensor=net_out_lig)
        eps_t = net_out_lig
        x_pred = 1. / alpha_t[batch_idx_lig] * (zt - sigma_t[batch_idx_lig] * eps_t)
        return x_pred