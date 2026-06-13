# import torch
# from repo.utils.molecule.constants import get_atomic_number_from_index, is_aromatic_from_index
# from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg, obabel_recover_bond

# def translate(result, translation):
#     result_pos = result[0].cpu()
#     result_pos += translation.cpu()
#     return [result_pos] + [result[k+1] for k in range(len(result) - 1)]


# def split_batch_into_samples(batch, mode='add_aromatic'):
#     batch_idx = batch[-1]
#     if batch_idx.numel() == 0:
#         return []
#     B = batch_idx.max() + 1
#     batch_split = []
#     for i in range(B):
#         idx = (batch_idx == i)
#         sample = {}
#         sample['pos'] = batch[0].cpu()[idx].tolist()
#         sample['type'] = batch[1].cpu()[idx].numpy()
#         if len(sample['type'].shape) == 2:
#             sample['type'] = sample['type'].argmax(axis=-1)
#         sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
#         sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
#         batch_split.append(sample)
#     return batch_split



# def reconstruct_save_mol(traj_batch, batch, save_dir, now_count, args, config):

#     count = now_count
#     if config.sampling.translate:
#         result_batch = translate(traj_batch[0], batch.protein_translation[:1])
#     else:
#         result_batch = traj_batch[0]      

#     result_split = split_batch_into_samples(result_batch, mode=config.mode)
    
#     for result in result_split:
#         try:
#             try:
#                 mol = reconstruct_mol(result['pos'], 
#                                         result['atom'], 
#                                         result['aromatic'], 
#                                         basic_mode=config.reconstruct.basic_mode)
#             except:
#                 mol = obabel_recover_bond(result['pos'], 
#                                             result['atom'])
                
#             mol, success = evaluate_validity(mol, args.threshold, args.threshold_ratio)
#             if success:
#                 if count >= 100:
#                     break
#                 count += 1
#                 save_mol(mol, os.path.join(save_dir, 'sample_%04d.sdf' % count))
#         except:
#             continue