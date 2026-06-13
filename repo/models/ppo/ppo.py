import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add
from torch.distributions import Categorical
from repo.models.ppo.reward import Reward, ConstantReward
from repo.models.ppo.policy import Actor, Critic, ConstantActor, ConstantCritic

def expand_x_insert_per_batch(x, batch_idx, gen_num, insert_mode='zero'):
    device = x.device
    B = gen_num.shape[0]

    x_parts = []
    batch_idx_parts = []

    for b in range(B):
        # 原 batch b 的行
        mask = (batch_idx == b)
        x_b = x[mask]
        batch_b = batch_idx[mask]

        # 要追加的行
        n_add = gen_num[b].item()
        if n_add > 0:
            if insert_mode == 'zero':
                zeros_b = torch.zeros(n_add, x.size(1), device=device)
            elif insert_mode == 'gauss':
                zeros_b = torch.randn((n_add, x.size(1)), device=device)
            elif insert_mode == 'uniform':
                uniform_dist = Categorical(logits=torch.ones_like(x))
                zeros_b = uniform_dist.sample()
                zeros_b = F.one_hot(zeros_b, num_classes=x.shape[-1])
            elif insert_mode == 'absorbing':
                zeros_b = torch.zeros(n_add, x.size(1), device=device)
                zeros_b[:, 0] = 1.
            else:
                raise ValueError
            batch_add = torch.full((n_add,), b, device=device, dtype=batch_idx.dtype)
        else:
            zeros_b = torch.empty(0, x.size(1), device=device)
            batch_add = torch.empty(0, dtype=batch_idx.dtype, device=device)

        # 拼 batch b 的结果（原 + 新）
        x_parts.append(torch.cat([x_b, zeros_b], dim=0))
        batch_idx_parts.append(torch.cat([batch_b, batch_add], dim=0))

    # 全部 batch 串起来
    x_new = torch.cat(x_parts, dim=0)
    batch_idx_new = torch.cat(batch_idx_parts, dim=0)

    return x_new, batch_idx_new

def sample_act(prob, batch_idx=None):
    dist = Categorical(probs=prob)
    act = dist.sample()
    act_log = dist.log_prob(act)
    if batch_idx == None:
        act_logits = act_log
    else:
        act_logits = scatter_add(act_log, batch_idx, dim=-1)
    return act, act_logits, dist

class PPOTrainer(nn.Module):
    def __init__(self, actor: Actor, critic: Critic, reward: Reward, 
                 sampler, discrete_type='gauss'):
        super(PPOTrainer, self).__init__()

        self.actor = actor
        self.critic = critic
        self.reward = reward
        self.sampler = sampler
        self.clip_epsilon = 0.2
        self.discrete_type = discrete_type

    def ppo_update(self, batch, old_actor):

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

        if v_lig_0.ndim == 1:
            v_lig_0 = F.one_hot(v_lig_0, num_classes=self.sampler.num_classes).float()

        pred_values, _ = self.critic(x_rec=x_rec_0, x_lig=x_lig_0, 
                                  h_rec=v_rec_0, h_lig=v_lig_0,
                                  batch_idx_rec=batch_idx_rec, 
                                  batch_idx_lig=batch_idx_lig)

        mask_prob, gen_prob = self.actor(x_rec=x_rec_0, x_lig=x_lig_0, 
                                         h_rec=v_rec_0, h_lig=v_lig_0,
                                         batch_idx_rec=batch_idx_rec, 
                                         batch_idx_lig=batch_idx_lig)
        
        mask_act, mask_act_logits, mask_dist = sample_act(mask_prob, batch_idx_lig)
        gen_act, gen_act_logits, gen_dist = sample_act(gen_prob)
        act_logits = mask_act_logits + gen_act_logits

        old_mask_prob, old_gen_prob = old_actor(x_rec=x_rec_0, x_lig=x_lig_0, 
                                                h_rec=v_rec_0, h_lig=v_lig_0,
                                                batch_idx_rec=batch_idx_rec, 
                                                batch_idx_lig=batch_idx_lig)
        old_mask_dist = Categorical(probs=old_mask_prob)
        old_mask_act_logits = old_mask_dist.log_prob(mask_act)
        old_mask_act_logits = scatter_add(old_mask_act_logits, batch_idx_lig, dim=0)
        old_gen_dist = Categorical(probs=old_gen_prob)
        old_gen_act_logits = old_gen_dist.log_prob(gen_act)
        old_act_logits = old_mask_act_logits + old_gen_act_logits

        fix_mask = (mask_act==0).bool()
        x_lig_new, batch_idx_lig_new = expand_x_insert_per_batch(
            x_lig_0[fix_mask], gen_num=gen_act,
            batch_idx=batch_idx_lig[fix_mask], insert_mode='gauss')
        v_lig_new, batch_idx_lig_new = expand_x_insert_per_batch(
            v_lig_0[fix_mask], gen_num=gen_act,
            batch_idx=batch_idx_lig[fix_mask], insert_mode=self.discrete_type)
        fix_mask_new, batch_idx_lig_new = expand_x_insert_per_batch(
            fix_mask[fix_mask][:, None], gen_num=gen_act,
            batch_idx=batch_idx_lig[fix_mask], insert_mode='zero')
        fix_mask_new = fix_mask_new[:, 0]
        lig_flag_new = torch.ones_like(batch_idx_lig_new)

        batch['ligand_pos'] = x_lig_new
        batch['ligand_atom_type'] = v_lig_new
        batch['ligand_lig_flag'] = lig_flag_new
        batch['ligand_gen_flag'] = ~fix_mask_new.bool()
        batch['ligand_element_batch'] = batch_idx_lig_new

        ts = list(reversed(range(0, 1000, 1000//1000)))
        traj_batch = self.sampler.dr_slover(batch, ts=ts, show=False)
        reward_v = self.reward.reward(traj_batch, batch)
        reward_v = reward_v[:, None].to(x_lig_0.device)
        advantages = reward_v - pred_values

        ratio = torch.exp(act_logits - old_act_logits)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        entropy_loss = scatter_add(mask_dist.entropy(), batch_idx_lig, dim=0) + gen_dist.entropy()
        # entropy_loss = scatter_add(entropy_loss, batch_idx_lig, dim=0)
        entropy_loss = entropy_loss.mean()

        actor_loss = ppo_loss - 0.05 * entropy_loss
        critic_loss = (pred_values - reward_v).pow(2).mean()

        loss_dict = {'ppo': ppo_loss, 'entropy': entropy_loss, 'actor': actor_loss,
                     'critic': critic_loss, 'reward': reward_v.mean()}
        
        return loss_dict


class ConstantPPOTrainer(nn.Module):
    def __init__(self, actor: ConstantActor, critic: ConstantActor, reward: ConstantReward, 
                 sampler, discrete_type='gauss'):
        super(ConstantPPOTrainer, self).__init__()

        self.actor = actor
        self.critic = critic
        self.reward = reward
        self.sampler = sampler
        self.clip_epsilon = 0.2
        self.discrete_type = discrete_type

    def ppo_update(self, batch, old_actor):

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

        if v_lig_0.ndim == 1:
            v_lig_0 = F.one_hot(v_lig_0, num_classes=self.sampler.num_classes).float()

        pred_values = self.critic(x_rec=x_rec_0, x_lig=x_lig_0, 
                                  h_rec=v_rec_0, h_lig=v_lig_0,
                                  batch_idx_rec=batch_idx_rec, 
                                  batch_idx_lig=batch_idx_lig)

        mask_prob = self.actor(x_rec=x_rec_0, x_lig=x_lig_0, 
                               h_rec=v_rec_0, h_lig=v_lig_0,
                               batch_idx_rec=batch_idx_rec, 
                               batch_idx_lig=batch_idx_lig)
        
        mask_act, mask_act_logits, mask_dist = sample_act(mask_prob, batch_idx_lig)
        act_logits = mask_act_logits
        gen_num = scatter_add(mask_act, batch_idx_lig, dim=0)

        old_mask_prob = old_actor(x_rec=x_rec_0, x_lig=x_lig_0, 
                                  h_rec=v_rec_0, h_lig=v_lig_0,
                                  batch_idx_rec=batch_idx_rec, 
                                  batch_idx_lig=batch_idx_lig)
        old_mask_dist = Categorical(probs=old_mask_prob)
        old_mask_act_logits = old_mask_dist.log_prob(mask_act)
        old_mask_act_logits = scatter_add(old_mask_act_logits, batch_idx_lig, dim=0)
        old_act_logits = old_mask_act_logits

        fix_mask = (mask_act==0).bool()
        x_lig_new, batch_idx_lig_new = expand_x_insert_per_batch(
            x_lig_0[fix_mask], gen_num=gen_num,
            batch_idx=batch_idx_lig[fix_mask], insert_mode='gauss')
        v_lig_new, batch_idx_lig_new = expand_x_insert_per_batch(
            v_lig_0[fix_mask], gen_num=gen_num,
            batch_idx=batch_idx_lig[fix_mask], insert_mode=self.discrete_type)
        fix_mask_new, batch_idx_lig_new = expand_x_insert_per_batch(
            fix_mask[fix_mask][:, None], gen_num=gen_num,
            batch_idx=batch_idx_lig[fix_mask], insert_mode='zero')
        fix_mask_new = fix_mask_new[:, 0]
        lig_flag_new = torch.ones_like(batch_idx_lig_new)

        batch['ligand_pos'] = x_lig_new
        batch['ligand_atom_type'] = v_lig_new
        batch['ligand_lig_flag'] = lig_flag_new
        batch['ligand_gen_flag'] = ~fix_mask_new.bool()
        batch['ligand_element_batch'] = batch_idx_lig_new

        ts = list(reversed(range(0, 1000, 1000//1000)))
        traj_batch = self.sampler.dr_slover(batch, ts=ts, show=False)
        reward_v, score, rmpg, success_rate = self.reward.reward(traj_batch, batch)
        reward_v = reward_v[:, None].to(x_lig_0.device)
        advantages = reward_v - pred_values

        ratio = torch.exp(act_logits - old_act_logits)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()

        entropy_loss = scatter_add(mask_dist.entropy(), batch_idx_lig, dim=0)
        # entropy_loss = scatter_add(entropy_loss, batch_idx_lig, dim=0)
        entropy_loss = entropy_loss.mean()

        actor_loss = ppo_loss - 0.0 * entropy_loss
        critic_loss = (pred_values - reward_v).pow(2).mean()

        loss_dict = {'ppo': ppo_loss, 'entropy': entropy_loss, 'actor': actor_loss,
                     'critic': critic_loss, 'reward': reward_v.mean(), 
                     'vina dock': score, 'rmpg': rmpg, 'rate': success_rate}
        
        return loss_dict
    
    def evaluate(self, batch):

        return 










