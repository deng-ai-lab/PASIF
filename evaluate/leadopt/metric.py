import os, sys
import pandas as pd
import numpy as np

def weighted_average(dataframe, value, weight=None):
    val = dataframe[value]
    if weight is None:
        return val.mean()
    else:
        wt = dataframe[weight]
        return (val * wt).sum() / wt.sum()

if __name__ == '__main__':

    eval_root = './results/scaffold/targetdiff/pretrain-dr_slover'
    pockets = os.listdir(eval_root)
    pockets = sorted(pockets)
    pockets_num = len(pockets)
    res_dict = {'pocket': [], 'vina score': [], 'vina score imp': [], 'vina min':[], 'vina min imp': [], 
                'vina dock': [], 'vina dock imp': [], 'vina dock MPBG': [], 'vina dock IBE': [], 
                'qed': [], 'logp': [], 'sa': [], 'lpsk': [], 'num': [], 'prop corr': []}
    no_success_num = 0
    pred_label = [[], []]
    for pocket in pockets:
        pocket_path = os.path.join(eval_root, pocket)
        if not os.path.isdir(pocket_path):
            continue
        tmps = os.listdir(pocket_path)
        tmps = sorted(tmps)
        if len(tmps) > 1:
            pockets_num += len(tmps) - 1
            print(f'Warning: more than one result in {pocket_path}')
        for tmp in tmps:
            tmp_path = os.path.join(pocket_path, tmp)
            if os.path.isdir(tmp_path):
                csv_path = os.path.join(tmp_path, 'molecule_properties.csv')
                prop_pred_path = os.path.join(tmp_path, 'qed_prop.csv')
                if os.path.exists(csv_path) is False:
                    print(f'Warning: no csv file in {csv_path}')
                    continue
                df = pd.read_csv(csv_path)
                if len(df) <= 1:
                    no_success_num += 1
                    print(f'Warning: no success molecule in {tmp_path}')
                    continue
                ref_idx = '/'.join(tmp_path.split('/')[-2:])
                ref_v = df.iloc[-1]

                if os.path.exists(prop_pred_path):
                    prop_df = pd.read_csv(prop_pred_path)
                    pred = prop_df['prop pred'].values
                    label = prop_df['qed'].values
                    rou = np.corrcoef(pred, label)[0, 1].item()
                    pred_label[0].extend(pred)
                    pred_label[1].extend(label)
                    res_dict['prop corr'].append(rou)
                else:
                    res_dict['prop corr'].append(1)
                
                num = min(len(df) - 1, 100)
                res_dict['pocket'].append(pocket)
                res_dict['num'].append(num)
                res_dict['vina score'].append(df['vina_score_result'].iloc[:num].mean())
                res_dict['vina score imp'].append(np.sum(df['vina_score_result'].values[:num] < ref_v['vina_score_result'])/num * 100)
                res_dict['vina min'].append(df['vina_min_result'].iloc[:num].mean())
                res_dict['vina min imp'].append(np.sum(df['vina_min_result'].values[:num] < ref_v['vina_min_result'])/num * 100)
                res_dict['vina dock IBE'].append(df['lbe_result'].iloc[:num].mean())
                res_dict['vina dock'].append(df['vina_dock_result'].iloc[:num].mean())
                res_dict['vina dock imp'].append(np.sum(df['vina_dock_result'].values[:num] < ref_v['vina_dock_result'])/num * 100)
                res_dict['vina dock MPBG'].append(np.mean((df['vina_dock_result'].values[:num] - ref_v['vina_dock_result'])
                                                        /ref_v['vina_dock_result']) * 100)
                res_dict['qed'].append(df['qed'].iloc[:num].mean())
                res_dict['logp'].append(df['logp'].iloc[:num].mean())
                res_dict['sa'].append(df['sa'].iloc[:num].mean())
                res_dict['lpsk'].append(df['lipinski'].iloc[:num].mean())
    res_df = pd.DataFrame(res_dict)
    weight = None
    if len(pred_label[0]) == 0:
        final_rou = 1
    else:
        final_rou = np.corrcoef(np.array(pred_label[0]), np.array(pred_label[1]))[0, 1].item()
    final_df = pd.DataFrame({'pocket': ['final'], 
                             'vina score': [weighted_average(res_df, 'vina score', weight)],
                             'vina score imp': [weighted_average(res_df, 'vina score imp', weight)],
                             'vina min': [weighted_average(res_df, 'vina min', weight)],
                             'vina min imp': [weighted_average(res_df, 'vina min imp', weight)],
                             'vina dock': [weighted_average(res_df, 'vina dock', weight)],
                             'vina dock imp': [weighted_average(res_df, 'vina dock imp', weight)],
                             'vina dock MPBG': [weighted_average(res_df, 'vina dock MPBG', weight)],
                             'vina dock IBE': [weighted_average(res_df, 'vina dock IBE', weight)],
                             'qed': [weighted_average(res_df, 'qed', weight)],
                             'logp': [weighted_average(res_df, 'logp', weight)],
                             'sa': [weighted_average(res_df, 'sa', weight)],
                             'lpsk': [weighted_average(res_df, 'lpsk', weight)],
                             'num': [res_df['num'].sum()/pockets_num], 
                             'prop corr': [final_rou]})
    res_df = pd.concat([res_df, final_df], ignore_index=True)
    decimal_places = {
                      'vina score': 2, 'vina score imp': 2, 'vina min': 2, 'vina min imp': 2, 'vina dock': 2,
                        'vina dock imp': 2, 'vina dock MPBG': 2, 'vina dock IBE': 4, 'qed': 2,
                        'logp': 2, 'sa': 2, 'lpsk': 2, 'num': 2, 'prop corr': 2
                     }
    for col, decimals in decimal_places.items():
        res_df[col] = res_df[col].round(decimals)
    res_df.to_csv(os.path.join(eval_root, 'summary.csv'), index=False, )