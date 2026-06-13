import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import argparse
import copy
import json
import subprocess
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader
import torch
from torch_scatter import scatter_mean
from repo.datasets.pl import get_pl_dataset
from repo.models import get_model
from repo.utils.misc import *
from repo.utils.molecule.constants import *
from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg, obabel_recover_bond
from repo.utils.data import recursive_to
from repo.models.classifier.classifier import PropPredictor
from repo.modules.e3nn.gvptransformer import GVPTransformer
from repo.models.diffusion.sampler import Sampler

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def split_batch_into_samples(batch, mode='add_aromatic'):
    batch_idx = batch[-1]
    if batch_idx.numel() == 0:
        return []
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos'] = batch[0].cpu()[idx].tolist()
        sample['type'] = batch[1].cpu()[idx].numpy()
        if len(sample['type'].shape) == 2:
            sample['type'] = sample['type'].argmax(axis=-1)
        sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
        sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
        batch_split.append(sample)
    return batch_split

def split_batch_into_samples_fg(batch, mode=None):
    batch_idx = batch[-1]
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos_center'] = batch[0].cpu()[idx].tolist()
        sample['fg_type'] = batch[1].cpu()[idx].numpy()
        if len(sample['fg_type'].shape) == 2:
            sample['fg_type'] = sample['fg_type'].argmax(axis=-1)
        sample['orientation'] = batch[2].cpu()[idx].numpy()
        batch_split.append(sample)
    return batch_split

def translate(result, translation):
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

parma_dict = {'ppb':     {'model_type': 'reg', 'ckp': '5000.pt', 'opt': 0.8},
              'oatp1b3': {'model_type': 'cls', 'ckp': '150.pt',  'opt': 0.2},
              'oatp1b1': {'model_type': 'cls', 'ckp': '400.pt',  'opt': 0.2},
              'cyp2c19_inh': {'model_type': 'cls', 'ckp': '5000.pt',  'opt': 0.2},}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--prop', type=str, default='oatp1b3')
    parser.add_argument('--config', type=str, default='./configs/denovo/test/diffbp.yml')
    parser.add_argument('--out_root', type=str, default='./results/admet/')
    parser.add_argument('--tag', type=str, default='pretrain')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--checkpoint', type=str, default='pretrained')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=float, default=0.6)

    args = parser.parse_args()
    model_type = parma_dict[args.prop]['model_type']
    pt_name = parma_dict[args.prop]['ckp']
    cls_ckpt_path = f'./logs/admet/{args.prop}/add_aromatic/self-train/checkpoints/{pt_name}'
 
    # Load configs
    config, config_name = load_config(args.config)
    if args.checkpoint is not None:
        config.model.checkpoint = os.path.join(
            "/".join(config.model.checkpoint.split('/')[:4]), 
            args.tag, 
            'checkpoints',
            args.checkpoint + '.pt'
        )

    if 'fg' not in config.model.type:
        from repo.utils.configuration import set_num_atom_type, set_num_bond_type
        set_num_atom_type(config)
        set_num_bond_type(config)
    else:
        from repo.utils.configuration import set_num_fg_type
        set_num_fg_type(config)

    seed_all(args.seed if args.seed is not None else config.sampling.seed)

    # Testset 
    datasets = get_pl_dataset(config.data.test)
    dataset = datasets['test']

    dr = os.path.join(args.out_root, config_name)

    if not os.path.exists(dr):

        os.makedirs(dr, exist_ok=True)

    mark = 0
    
    log_dir = get_new_log_dir(dr, prefix='', tag=args.tag if config.model.type != 'difffg' else 'context')

    logger = get_logger('sample', log_dir)

    # Load checkpoint and model
    logger.info('Loading model config and checkpoints: %s' % (config.model.checkpoint))
    ckpt = torch.load(config.model.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    cls_ckpt = torch.load(cls_ckpt_path, map_location='cpu')
    classifier = PropPredictor(cls_ckpt['config']['model'], model_type=model_type).to(args.device)
    classifier.load_state_dict(cls_ckpt['model'])
    logger.info(str(lsd))
    logger.info(args.out_root)

    for i in range(mark, len(dataset)):
        
        
        args.index = i
        get_structure = lambda: dataset[args.index]
        get_raw_structure = lambda: dataset.dataset.get_raw(dataset.indices[args.index])

        # Logging
        raw_strcuture_ = get_raw_structure()

        structure_id = raw_strcuture_['entry'][0][:-4]

        save_dir = os.path.join(log_dir+'-'+args.prop, '%s' % (structure_id))
        os.makedirs(save_dir, exist_ok=True)

        logger.info('Data ID: %s' % structure_id)
        
        data_native = {'entry': raw_strcuture_['entry']}

        all_files = []
        for f in os.listdir(save_dir):
            if f.split('_')[0]=='sample':
                all_files.append(f)
        all_files = sorted(all_files)
        cfg_num_samples = args.num_samples
        if len(all_files) > 0:
            samples_idx = all_files[-1]
            samples_idx = samples_idx.split('_')[-1]
            samples_idx = int(samples_idx.split('.')[0])
            num_samples = cfg_num_samples - samples_idx
            if num_samples <= 0:
                logger.info('Already generated enough samples for %s, skipping...' % structure_id)
                continue
            num_more = max(num_samples, 1)
        else:
            samples_idx = 0
            num_more = cfg_num_samples
        data_list_repeat = [get_structure() for _ in range(num_more)]
        batch_size = args.batch_size if config.data.get('batch_size', None) is None else config.data.batch_size
        
        loader = DataLoader(PygDatasetFromList(data_list_repeat), 
                            batch_size=batch_size, 
                            shuffle=False,
                            follow_batch=config.data.get('follow_batch', []))
        
        count = samples_idx
        mol_part_list = []
        pred_prop = None
        for batch in tqdm(loader, desc=structure_id, dynamic_ncols=True):

            try:
                batch = batch.to(args.device)
            except:
                batch = recursive_to(batch, args.device)
            
            ts = list(reversed(range(0, 1000, 1000//1000)))
            traj_batch, pred_prop = model.dr_guid(batch, ts=ts, classifier=classifier, just_lig=True, 
                                                    opt_value=parma_dict[args.prop]['opt'])

            if len(traj_batch) == 0: 
                logger.warning('No samples generated for %s, skipping...' % structure_id)
                continue

            if config.sampling.translate:
                result_batch = translate(traj_batch[0], batch.protein_translation[:1])
            else:
                result_batch = traj_batch[0]      

            if 'fg' in config.model.type:
                result_split = split_batch_into_samples_fg(result_batch, mode=config.mode) 
            else:
                result_split = split_batch_into_samples(result_batch, mode=config.mode)

            if config.get('reconstruct', None) is not None:
                
                for i, result in enumerate(result_split):
                    try:
                        try:
                            mol = reconstruct_mol(result['pos'], 
                                                  result['atom'], 
                                                  result['aromatic'], 
                                                  basic_mode=config.reconstruct.basic_mode)
                        except:
                            mol = obabel_recover_bond(result['pos'], 
                                                      result['atom'])
                            
                        mol, success = evaluate_validity(mol, args.threshold, args.threshold_ratio)
                        if success:
                            if count >= args.num_samples:
                                break
                            count += 1
                            if pred_prop is not None:
                                data = {'pos': np.array(result['pos']),
                                        'atom': np.array(result['atom']),
                                        'entry': data_native['entry'],
                                        'prop_pred': pred_prop[i].numpy().item()}
                            else:
                                data = {'pos': np.array(result['pos']),
                                        'atom': np.array(result['atom']),
                                        'entry': data_native['entry'],
                                        'prop_pred': None}
                            torch.save(data, os.path.join(save_dir, 'sample_%04d.pt' % count))
                            save_mol(mol, os.path.join(save_dir, 'sample_%04d.sdf' % count))
                    except:
                        continue
            
            elif config.get('fg2mol', None) is not None:
                for result in result_split:
                    part_mol = atom_from_fg(result['pos_center'],
                                            result['orientation'],
                                            result['fg_type'])
                    mol_part_list.append(part_mol)
        
        if config.get('fg2mol', None) is not None:
            torch.save(mol_part_list, os.path.join(save_dir, 'gen_ctx_pool_%04d.pt' % len(mol_part_list)))
            torch.save(mol_part_list, os.path.join(save_dir, 'gen_ctx_pool_raw.pt'))

if __name__ == '__main__':
    main()
