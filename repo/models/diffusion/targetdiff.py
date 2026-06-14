from torch import nn 
import torch
import math
import copy
from .diffusion_scheduler import CTNVPScheduler, TypeVPScheduler
from repo.modules.e3nn import get_e3_gnn
from repo.modules.context_emb import get_context_embedder
from .._base import register_model
from repo.utils.molecule.constants import *
from repo.utils.protein.constants import *
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from torch_scatter import scatter_mean, scatter_add
from repo.modules.common import compose_context, get_dict_mean
from tqdm.auto import tqdm
from ._base import BaseDiff
from repo.models.charge_density.utils import prepare_input, prepare_input_given_grid


@register_model('targetdiff')
class TargetDiff(BaseDiff):

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.cfg = cfg
        pos_scheduler_cfg = cfg.generator.pos_schedule
        self.num_classes = cfg.num_atomtype
        
        self.pos_scheduler = CTNVPScheduler(self.num_diffusion_timesteps, 
                                            beta_start = pos_scheduler_cfg.beta_start, 
                                            beta_end = pos_scheduler_cfg.beta_end, 
                                            type = pos_scheduler_cfg.type)
        
        atom_scheduler_cfg = cfg.generator.atom_schedule
        self.type_scheduler = TypeVPScheduler(self.num_diffusion_timesteps,
                                              num_classes = self.num_classes,
                                              type = atom_scheduler_cfg.type,
                                              cosine_s = atom_scheduler_cfg.cosine_s)
        
        cfg.embedder.num_atomtype = cfg.num_atomtype
        self.context_embedder = get_context_embedder(cfg.embedder)
        
        self.denoiser = get_e3_gnn(cfg.encoder, num_classes = self.num_classes)


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
            t = self.sample_time(B, device = x_lig_0.device)
            return self.get_loss(x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                                 lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                                 gen_flag_lig, gen_flag_rec, t)
        
        else:
            loss_dicts = []
            results = []
            eval_times = np.linspace(0, 
                                     self.num_diffusion_timesteps - 1, 
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

    def get_loss(self, x_lig_0, x_rec_0, v_lig_0, v_rec_0, aa_rec_0,
                  lig_flag, rec_flag, batch_idx_lig, batch_idx_rec, 
                  gen_flag_lig, gen_flag_rec, t):
        
        if self.denoise_structure:
            x_lig_t, _ = self.pos_scheduler.forward_add_noise(x_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            x_lig_t = x_lig_0

        if self.denoise_atom:
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v_lig_0, t, batch_idx_lig, gen_flag_lig)
        else:
            c_lig_t = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        x_lig_t, x_rec_t, h_lig_t, h_rec_t = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                  batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
        context_composed, batch_idx, _ = compose_context({'x': x_lig_t, 'h': h_lig_t, 'gen_flag': gen_flag_lig, 'lig_flag': lig_flag},
                                                         {'x': x_rec_t, 'h': h_rec_t, 'gen_flag': gen_flag_rec, 'lig_flag': rec_flag},
                                                          batch_idx_lig, batch_idx_rec)
        
        x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)
        x_lig_pred = x[context_composed['lig_flag']]
        c_lig_pred = v[context_composed['lig_flag']]

        if self.denoise_structure:    
            loss_pos, pos_info = self.pos_scheduler.get_loss(x_lig_pred, x_lig_0, x_lig_t, t, 
                                                             gen_flag_lig, batch_idx_lig, 
                                                             type='denoise')
        else:
            loss_pos, pos_info = torch.tensor(0).float(), {}
        if self.denoise_atom:
            loss_atom, atom_info = self.type_scheduler.get_loss(c_lig_pred, v_lig_0, v_lig_t, t, 
                                                                gen_flag_lig, batch_idx_lig, 
                                                                pred_logit=True)
        else:
            loss_atom, atom_info = torch.tensor(0).float(), {}

        results = {}
        results.update(pos_info)
        results.update(atom_info)

        return {'pos': loss_pos, 'atom': loss_atom}, results
    
    def softmax_jacobian(self, z: torch.Tensor, temp) -> torch.Tensor:
        s = F.softmax(z/temp, dim=-1)
        B, N = s.shape
        # diag(s) -> (B, N, N)
        diag_s = torch.diag_embed(s)

        # outer product s s^T -> (B, N, N)
        outer_s = s.unsqueeze(2) @ s.unsqueeze(1)

        return (diag_s - outer_s) / temp

    def sample(self, batch):
        torch.set_grad_enabled(False)
        self.eval()
        x_lig_in = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
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
        c_lig_in = F.one_hot(v_lig_in, num_classes = self.num_classes).float()

        time_seq = list(reversed(range(0, self.num_diffusion_timesteps)))
        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        traj = {self.num_diffusion_timesteps - 1: (x_lig_in, c_lig_in, batch_idx_lig)}

        for t_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig, c_lig, _ = traj[t_idx]

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=x_lig_in.device)

            x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig, x_rec_0, c_lig, v_rec_0, aa_rec_0, 
                                                              batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
        
            context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': gen_flag_lig, 'lig_flag':lig_flag},
                                                             {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                             batch_idx_lig, batch_idx_rec)
            
            x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

            x_lig_out = x[context_composed['lig_flag']]
            c_lig_out = v[context_composed['lig_flag']]

            if self.denoise_structure:    
                x_lig_next = self.pos_scheduler.backward_remove_noise(x_lig_out, x_lig, t, 
                                                                      batch_idx_lig, gen_flag_lig, 
                                                                      type='denoise')
            else:
                x_lig_next = x_lig
                
            if self.denoise_atom:
                c_lig_next, _ = self.type_scheduler.backward_remove_noise(c_lig_out, c_lig, t, 
                                                                          batch_idx_lig, gen_flag_lig, 
                                                                          pred_logit=True)
            else:
                c_lig_next = c_lig
            
            traj[t_idx - 1] = (x_lig_next, c_lig_next, batch_idx_lig)
            traj[t_idx] = tuple(x.cpu() for x in traj[t_idx]) 

        return traj
    
    def inpaint(self, batch, resamples):
        x_lig_in = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
        x_rec_0 = batch['protein_pos']
        v_rec_0 = batch['protein_atom_feature']
        aa_rec_0 = batch['protein_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        self.type_scheduler.Kt = self.type_scheduler.Kt.to(x_rec_0.device)

        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()
        c_lig_in = F.one_hot(v_lig_in, num_classes = self.num_classes).float()

        time_steps = int(2 * self.num_diffusion_timesteps / resamples)
        true_t = lambda t: (t / time_steps * self.num_diffusion_timesteps)
        time_seq = list(reversed(range(0, time_steps)))

        N_lig, _ = x_lig_in.shape
        N_rec, _ = x_rec_0.shape
        B = batch_idx_lig.max() + 1

        unif_dist = Categorical(probs=torch.ones_like(c_lig_in)/self.num_classes)

        x_lig_init = torch.randn_like(x_lig_in)
        v_lig_init = unif_dist.sample()
        c_lig_init = F.one_hot(v_lig_init, num_classes = self.num_classes).float()
        
        diff_mask = torch.ones_like(gen_flag_lig)

        traj = {time_seq[0]+1: (x_lig_init, c_lig_init, batch_idx_lig)}

        for s_idx in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            x_lig_unknown, c_lig_unknown, _ = traj[s_idx+1]
            s = torch.full(size=(B,), fill_value=s_idx, dtype=torch.long, device=x_lig_in.device)
            t = s + 1
            t_diff = true_t(t).long()
            s_diff = true_t(s).long()
            re_i = 0
            while(re_i < resamples):
                
                x_lig_known, _ = self.pos_scheduler.forward_add_noise(x_lig_in, t_diff, batch_idx_lig, diff_mask)
                x_lig_t = (1 - gen_flag_lig[:, None].float()) * x_lig_known + \
                        gen_flag_lig[:, None].float() * x_lig_unknown
                
                c_lig_known, v_lig_known = self.type_scheduler.forward_add_noise(v_lig_in, t_diff, batch_idx_lig, diff_mask)
                c_lig_t = (1 - gen_flag_lig[:, None].float()) * c_lig_known + \
                        gen_flag_lig[:, None].float() * c_lig_unknown            

                with torch.no_grad():
                    x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                    batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t_diff)
                
                    context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': diff_mask, 'lig_flag':lig_flag},
                                                                    {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                    batch_idx_lig, batch_idx_rec)
                    
                    x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                    x_lig_out = x[context_composed['lig_flag']]
                    c_lig_out = v[context_composed['lig_flag']]

                    c_pred = F.softmax(c_lig_out, dim=-1)

                if self.denoise_structure:    
                    x_lig_next = self.pos_scheduler.sample_xs_given_x0_xt(x0=x_lig_out, xt=x_lig_t, s=s_diff, t=t_diff, 
                                                                          batch=batch_idx_lig, gen_flag=diff_mask)
                    x_lig_unknown, _ = self.pos_scheduler.multi_step_add_noise(x_lig_next, t_diff, s_diff, 
                                                                                batch_idx_lig, diff_mask)
                else:
                    x_lig_next = x_lig_t
                    x_lig_unknown = x_lig_t
                    
                if self.denoise_atom:
                    c_lig_next, _ = self.type_scheduler.sample_xs_given_xt_x0(c_pred, c_lig_t, t_diff, s_diff, 
                                                                              batch_idx_lig, diff_mask)
                    c_lig_unknown, _ = self.type_scheduler.multi_step_add_noise(c_lig_next, t_diff, s_diff, 
                                                                                batch_idx_lig, diff_mask)
                else:
                    c_lig_next = c_lig_t
                    c_lig_unknown = c_lig_t

                re_i += 1
            
            traj[s_idx] = (x_lig_next, c_lig_next, batch_idx_lig)
            traj[s_idx+1] = tuple(x.cpu() for x in traj[s_idx+1])
        traj[s_idx] = tuple(x.cpu() for x in traj[s_idx])
        return traj

    def loss_noise_p0t(self, x_t, t, q0, p0t, batch_idx):
        """
        param:
            -x_t: (L, S)
            -t: (N, )
            -q0: (L, S)
            -p0t: (L, S)
            -batch_idx: (L, )
        """
        # L = batch_idx.shape[0]
        L, S = x_t.shape
        x_t_idx = torch.argmax(x_t, dim=-1) # (L,)

        pt0 = self.type_scheduler.Kt[t[0]].to(x_t.device)  # (S, S)
        pt0 = pt0[None, ...].repeat(L, 1, 1) # (L, S, S)
        # pt0_x = pt0.gather(-1, x_t_idx.view(L, 1, 1).expand(-1, S, -1))[:, :, 0]
        pt0_x = pt0[torch.arange(L), :, x_t_idx]  # (L, S)

        qt = self.type_scheduler.qt(q0, pt0, batch_idx)  # (L, S)
        qt_x = qt[torch.arange(L), x_t_idx][:, None]  # (L, 1)

        q0t = (q0[:, :, None] * pt0) / qt[:, None, :].clamp(min=0.001, max=0.999)  # (L, S, S)
        # q0t_x = q0t.gather(-1, x_t_idx.view(L, 1, 1).expand(-1, S, -1))[:, :, 0]  # (L, S)
        q0t_x = q0t[torch.arange(L), :, x_t_idx]  # (L, S)

        grad = pt0_x.clamp(min=1.e-3, max=0.999) * torch.log(q0t_x.clamp(min=1.e-3, max=0.999) / p0t.clamp(min=1.e-3, max=0.999)) / qt_x.clamp(min=1.e-3, max=0.999)
        # grad = pt0_x.clamp(min=1.e-3, max=0.999) * torch.log(q0t_x.clamp(min=1.e-3, max=0.999) / p0t.clamp(min=1.e-3, max=0.999))

        loss_out = (grad.detach() * q0).mean()

        return loss_out, grad
    
    def loss_noise_pt(self, x_t, t, q0, p0t, batch_idx):
        """
        param:
            -x_t: (L, S)
            -t: (N, )
            -q0: (L, S)
            -p0t: (L, S)
            -batch_idx: (L, )
        """
        L, S = x_t.shape
        x_t_idx = torch.argmax(x_t, dim=-1)

        pT_pt = self.type_scheduler.pT_pt(t=t[0], p0t=p0t, x=x_t, batch_idx=batch_idx)  # (L,)
        # pT_pt = pT_pt.clamp(min=1.e-3, max=0.999)[:, None]
        pT_pt = pT_pt[:, None] + 1.e-5

        pt0 = self.type_scheduler.Kt[t[0]].to(x_t.device)  # (S, S)
        pt0 = pt0[None, ...].repeat(L, 1, 1) # (L, S, S)
        qt = self.type_scheduler.qt(q0, pt0, batch_idx)  # (L, S)
        qt_x = qt[torch.arange(L), x_t_idx]  # (L,)
        qt_qT = qt_x / (1/S)
        # qt_qT = qt_qT.clamp(min=1.e-3, max=0.999)[:, None]
        qt_qT = qt_qT[:, None] + 1.e-5

        # grad = pt0[torch.arange(L), :, x_t_idx] * (torch.log(qt_qT) + torch.log(pT_pt)) + 1.  # (L, S)

        grad = pt0[torch.arange(L), :, x_t_idx] * (torch.log(qt_qT) + torch.log(pT_pt)) / qt_x[:, None] + 1.  # (L, S)


        loss_out = (grad.detach() * q0).mean()

        return loss_out, grad

    def dr_slover(self, batch, ts, init_noise=False):
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

        if gen_flag_lig.all().item():
            return []
        def w_t(s):
            if s < 0.3:
                return 0.1, 1.0
            elif s >= 0.3 and s <= 0.8:
                return 0.1, 1.0
            else:
                return 1.0, 0.1

        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()
        c_lig_0 = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        B = batch_idx_lig.max() + 1
        sigma_x0 = 0.0001  # variational dist var
        sigma_c0 = 0.001  # observe dist var
        
        unif_probs = torch.ones_like(c_lig_0) / c_lig_0.shape[-1]
        x_lig_unknown = torch.randn_like(x_lig_0)
        if init_noise:
            x_lig_in = x_lig_unknown
            c_lig_in = unif_probs
        else:
            x_lig_in = gen_flag_lig[:, None].float() * x_lig_unknown + \
                    (1. - gen_flag_lig[:, None].float()) * x_lig_0
            c_lig_in = gen_flag_lig[:, None].float() * unif_probs + \
                    (1. - gen_flag_lig[:, None].float()) * c_lig_0
        x_lig_mu = torch.autograd.Variable(x_lig_in, requires_grad=True)
        c_lig_mu = torch.autograd.Variable(c_lig_in, requires_grad=True)
        h_mask = 1. - (gen_flag_lig).float()[:, None]
        opt_mask = torch.ones_like(gen_flag_lig)

        optimizer = Adam([x_lig_mu, c_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

        traj = {ts[0]: (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())}

        temp = 0.5
        for t_idx in tqdm(ts, desc='sampling', total=len(ts)):

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=x_lig_in.device)

            if t_idx == ts[0]:
                c_lig_mu_norm = c_lig_mu
            else:
                c_lig_mu_norm = F.softmax(c_lig_mu/temp, dim=-1)
            
            noise_x0 = torch.randn_like(x_lig_mu)
            x0_pred = x_lig_mu + sigma_x0 * noise_x0
            c0_pred = c_lig_mu_norm
            dist = Categorical(probs=c0_pred)
            v0_pred = dist.sample()

            # att
            x_lig_t, pos_noise = self.pos_scheduler.forward_add_noise(x0_pred, t, batch_idx_lig, opt_mask)
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v0_pred, t, batch_idx_lig, opt_mask)

            with torch.no_grad():
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
            
                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': opt_mask, 'lig_flag':lig_flag},
                                                                {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_out = x[context_composed['lig_flag']]
                c_lig_out = v[context_composed['lig_flag']]

            pyx = (1 - sigma_c0) * c_lig_0 + sigma_c0 * unif_probs            
            
            grad_x_obs = h_mask * (x_lig_mu - x_lig_0)
            grad_c_obs = - h_mask * (torch.log(pyx.clamp(min=1.e-3, max=0.999)))
            grad_x_noise = (x_lig_mu - x_lig_out).detach()

            loss_noise_p0t, grad_p0t = self.loss_noise_p0t(c_lig_t, t, q0=c_lig_mu_norm, p0t=F.softmax(c_lig_out, dim=-1), batch_idx=batch_idx_lig)
            loss_noise_pt, grad_pt = self.loss_noise_pt(c_lig_t, t, q0=c_lig_mu_norm, p0t=F.softmax(c_lig_out, dim=-1), batch_idx=batch_idx_lig)
            grad_c_noise = grad_p0t + grad_pt

            grad_x = grad_x_obs + 0.25 * grad_x_noise
            grad_c_norm = grad_c_obs + 3.0 * grad_c_noise

            jacob = self.softmax_jacobian(c_lig_mu, temp)
            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]
            
            optimizer.zero_grad()
            x_lig_mu.grad = grad_x
            c_lig_mu.grad = grad_c
            optimizer.step()
            
            traj[t_idx - 1] = (x_lig_mu.detach().cpu(), F.softmax(c_lig_mu/temp, dim=-1).detach().cpu(), batch_idx_lig.detach().cpu())

        return traj
    
    def specify_guid(self, batch, ts, classifier):

        x_rec_0 = batch['protein_pos']
        x_off_0 = batch['offtarget_pos']
        x_lig_0 = batch['ligand_pos']
        v_rec_0 = batch['protein_atom_feature']
        v_off_0 = batch['offtarget_atom_feature']
        v_lig_0 = batch['ligand_atom_type']
        aa_rec_0 = batch['protein_aa_type']
        aa_off_0 = batch['offtarget_aa_type']
        lig_flag = batch['ligand_lig_flag']
        rec_flag = batch['protein_lig_flag']
        off_flag = batch['offtarget_lig_flag']
        gen_flag_lig = batch.get('ligand_gen_flag', lig_flag)
        batch_idx_lig = batch['ligand_element_batch']
        batch_idx_rec = batch['protein_element_batch']
        batch_idx_off = batch['offtarget_element_batch']
        gen_flag_rec = batch.get('protein_gen_flag', torch.zeros_like(rec_flag))

        aa_rec_0 = F.one_hot(aa_rec_0, num_classes = len(aa_name_number)).float()
        aa_off_0 = F.one_hot(aa_off_0, num_classes = len(aa_name_number)).float()
        c_lig_0 = F.one_hot(v_lig_0, num_classes = self.num_classes).float()

        B = batch_idx_lig.max() + 1
        sigma_x0 = 0.0001  # variational dist var
        
        unif_probs = torch.ones_like(c_lig_0) / c_lig_0.shape[-1]
        x_lig_unknown = torch.randn_like(x_lig_0)

        x_lig_in = gen_flag_lig[:, None].float() * x_lig_unknown + \
                (1. - gen_flag_lig[:, None].float()) * x_lig_0
        c_lig_in = gen_flag_lig[:, None].float() * unif_probs + \
                (1. - gen_flag_lig[:, None].float()) * c_lig_0
        x_lig_mu = torch.autograd.Variable(x_lig_in, requires_grad=True)
        c_lig_mu = torch.autograd.Variable(c_lig_in, requires_grad=True)
        opt_mask = torch.ones_like(gen_flag_lig)

        optimizer = Adam([x_lig_mu, c_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

        traj = {ts[0]: (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())}

        temp = 0.5
        for t_idx in tqdm(ts, desc='sampling', total=len(ts)):

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=x_lig_in.device)

            if t_idx == ts[0]:
                c_lig_mu_norm = c_lig_mu
            else:
                c_lig_mu_norm = F.softmax(c_lig_mu/temp, dim=-1)
            
            noise_x0 = torch.randn_like(x_lig_mu)
            x0_pred = x_lig_mu + sigma_x0 * noise_x0
            c0_pred = c_lig_mu_norm
            dist = Categorical(probs=c0_pred)
            v0_pred = dist.sample()

            x_lig_t, pos_noise = self.pos_scheduler.forward_add_noise(x0_pred, t, batch_idx_lig, opt_mask)
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v0_pred, t, batch_idx_lig, opt_mask)

            with torch.no_grad():
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
            
                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': opt_mask, 'lig_flag':lig_flag},
                                                                {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_out = x[context_composed['lig_flag']]
                c_lig_out = v[context_composed['lig_flag']]
            
            if t_idx <=1000:
                prop_pred = classifier.predict(x_rec=x_rec_0, x_lig=x_lig_mu, 
                                            h_rec=v_rec_0, h_lig=c_lig_mu_norm,
                                            batch_idx_rec=batch_idx_rec, batch_idx_lig=batch_idx_lig)
                prop_pred = prop_pred.clamp(min=0.0, max=1.0)

                prop_pred_off = classifier.predict(x_rec=x_off_0, x_lig=x_lig_mu, 
                                                h_rec=v_off_0, h_lig=c_lig_mu_norm,
                                                batch_idx_rec=batch_idx_off, batch_idx_lig=batch_idx_lig)
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
            
            grad_x_noise = (x_lig_mu - x_lig_out).detach()

            loss_noise_p0t, grad_p0t = self.loss_noise_p0t(c_lig_t, t, q0=c_lig_mu_norm, p0t=F.softmax(c_lig_out, dim=-1), batch_idx=batch_idx_lig)
            loss_noise_pt, grad_pt = self.loss_noise_pt(c_lig_t, t, q0=c_lig_mu_norm, p0t=F.softmax(c_lig_out, dim=-1), batch_idx=batch_idx_lig)
            grad_c_noise = grad_p0t + grad_pt

            jacob = self.softmax_jacobian(c_lig_mu, temp)

            grad_x = 0.25 * grad_x_noise + w * grad_prop_x + w * grad_propoff_x
            grad_c_norm = 3.0 * grad_c_noise + w * grad_prop_c + w * grad_propoff_c

            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]
            
            optimizer.zero_grad()
            x_lig_mu.grad = grad_x
            c_lig_mu.grad = grad_c
            optimizer.step()
            
            traj[t_idx - 1] = (x_lig_mu.detach().cpu(), F.softmax(c_lig_mu/temp, dim=-1).detach().cpu(), batch_idx_lig.detach().cpu())

        return traj

    def charge_density_guid(self, batch, ts, classifier, params, local=False):

        x_lig_in = batch['ligand_pos']
        v_lig_in = batch['ligand_atom_type']
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
        c_lig_in = F.one_hot(v_lig_in, num_classes = self.num_classes).float()

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

        sigma_x0 = 0.0001  # variational dist var

        x_lig_mu = torch.autograd.Variable(x_lig_in.detach().clone(), requires_grad=True)
        c_lig_mu = torch.autograd.Variable(c_lig_in.detach().clone(), requires_grad=True)
        opt_mask = torch.ones_like(gen_flag_lig)

        optimizer = Adam([x_lig_mu, c_lig_mu], lr=0.1, betas=(0.9, 0.99), weight_decay=0.0)

        traj = {self.num_diffusion_timesteps - 1: (x_lig_mu.detach().cpu(), c_lig_mu.detach().cpu(), batch_idx_lig.detach().cpu())}

        temp = 0.5
        loss_list = [[], [], []]
        grad_norm_list = [[], []]
        for t_idx in tqdm(ts, desc='sampling', total=len(ts)):
            x_lig, c_lig, _ = traj[t_idx]

            t = torch.full(size=(B,), fill_value=t_idx, dtype=torch.long, device=x_lig_in.device)

            if t_idx == ts[0]:
                c_lig_mu_norm = c_lig_mu
            else:
                c_lig_mu_norm = F.softmax(c_lig_mu/temp, dim=-1)
            
            noise_x0 = torch.randn_like(x_lig_mu)
            x0_pred = x_lig_mu + sigma_x0 * noise_x0
            c0_pred = c_lig_mu_norm
            dist = Categorical(probs=c0_pred)
            v0_pred = dist.sample()

            # att
            x_lig_t, pos_noise = self.pos_scheduler.forward_add_noise(x0_pred, t, batch_idx_lig, opt_mask)
            c_lig_t, v_lig_t = self.type_scheduler.forward_add_noise(v0_pred, t, batch_idx_lig, opt_mask)

            with torch.no_grad():
                x_lig, x_rec, h_lig, h_rec = self.context_embedder(x_lig_t, x_rec_0, c_lig_t, v_rec_0, aa_rec_0, 
                                                                batch_idx_lig, batch_idx_rec, lig_flag, rec_flag, t)
            
                context_composed, batch_idx, _ = compose_context({'x': x_lig, 'h': h_lig, 'gen_flag': opt_mask, 'lig_flag':lig_flag},
                                                                {'x': x_rec, 'h': h_rec, 'gen_flag': gen_flag_rec, 'lig_flag':rec_flag},
                                                                batch_idx_lig, batch_idx_rec)
                
                x, h, v = self.denoiser(batch_idx=batch_idx, **context_composed)

                x_lig_out = x[context_composed['lig_flag']]
                c_lig_out = v[context_composed['lig_flag']]
                c_pred = F.softmax(c_lig_out, dim=-1)

            if t_idx <=400 and t_idx >= 0:

                input_dict = copy.deepcopy(params)
                input_dict['grid_pos'] = gird_pos
                cls_batch = prepare_input_given_grid((x_lig_mu.detach().cpu(), 
                                           F.softmax(c_lig_mu/temp, dim=-1).detach().cpu(), 
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
                    high_rho[high_rho < 0.010] = -0.02
                    high_rho[high_rho >= 0.010] = 0.0
                    peak_rho[peak_rho < 0.05] = 0.0
                    peak_rho[peak_rho >= 0.05] = 0.02
                    loss_high =  - (high_rho * prop_pred).sum(-1).mean(0)
                    loss_peak = - (peak_rho * prop_pred).sum(-1).mean(0)
                    loss_prop = loss_high + 0.1 * loss_peak

                    w = 0.0 * t_idx/400 + 30.0 * (1-t_idx/400)

                else:
                    scale = tar_rho.sum(-1, keepdim=True) / prop_pred.sum(-1, keepdim=True)
                    prop_pred = scale * prop_pred
                    tar_rho = tar_rho.clamp(min=1.e-3, max=10)
                    prop_pred = prop_pred.clamp(min=1.e-3, max=10)
                    loss_shape = (prop_pred - tar_rho).abs().sum(-1) / tar_rho.abs().sum(-1)
                    loss_shape = loss_shape.mean()
                    loss_scale = (scale - 1.).abs().mean()
                    loss_prop = loss_shape + 0. * loss_scale

                    w = 25
                
                grad_prop_x = torch.autograd.grad(loss_prop, cls_batch['atom_xyz'], retain_graph=False)[0] * B
                grad_prop_x = grad_prop_x.reshape(-1, 3)[cls_batch['heavy_mask'].reshape(-1)]
                grad_prop_c = torch.zeros_like(c_lig_mu_norm)
                
                if torch.isnan(grad_prop_x).any().item() == True:
                    mask_nan = torch.isnan(grad_prop_x)
                    grad_prop_x[mask_nan] = 0.
                                
            else:
                prop_pred = None
                grad_prop_x = torch.zeros_like(x_lig_mu)
                grad_prop_c = torch.zeros_like(c_lig_mu_norm)
                w = 0.0

            grad_x_noise = (x_lig_mu - x_lig_out).detach()

            loss_noise_p0t, grad_p0t = self.loss_noise_p0t(c_lig_t, t, q0=c_lig_mu_norm, p0t=c_pred, batch_idx=batch_idx_lig)
            loss_noise_pt, grad_pt = self.loss_noise_pt(c_lig_t, t, q0=c_lig_mu_norm, p0t=c_pred, batch_idx=batch_idx_lig)
            grad_c_noise = grad_p0t + grad_pt

            grad_x = 0.25 * grad_x_noise + w * grad_prop_x
            grad_c_norm = 3.0 * grad_c_noise + w * grad_prop_c

            jacob = self.softmax_jacobian(c_lig_mu, temp)
            grad_c = torch.matmul(jacob.transpose(-1, -2), grad_c_norm[:, :, None])[:, :, 0]

            grad_norm_list[0].append(torch.norm(grad_x, dim=-1).mean().item())
            grad_norm_list[1].append(torch.norm(grad_c, dim=-1).mean().item())

            optimizer.zero_grad()
            x_lig_mu.grad = grad_x
            c_lig_mu.grad = grad_c
            optimizer.step()
            
            traj[t_idx - 1] = (x_lig_mu.detach().cpu(), F.softmax(c_lig_mu/temp, dim=-1).detach().cpu(), batch_idx_lig.detach().cpu())
        return traj
