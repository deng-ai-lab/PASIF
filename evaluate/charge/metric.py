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

    eval_root = './results/charge_local/diffsbdd/'

    pockets = os.listdir(eval_root)
    pockets = sorted(pockets)
    pockets_num = len(pockets)
    res_dict = {'pocket': [], 'q_value': [], 'num': []}
    no_success_num = 0
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
                csv_path = os.path.join(tmp_path, 'q_value.csv')
                if os.path.exists(csv_path) is False:
                    print(f'Warning: no csv file in {csv_path}')
                    continue
                df = pd.read_csv(csv_path)
                if len(df) <= 1:
                    no_success_num += 1
                    print(f'Warning: no success molecule in {tmp_path}')
                    continue

                num = min(len(df), 5)
                res_dict['pocket'].append(pocket)
                res_dict['q_value'].append(df['q'].iloc[:num].mean())
                res_dict['num'].append(num)
    res_df = pd.DataFrame(res_dict)
    weight = None
    final_df = pd.DataFrame({'pocket': ['final'], 
                             'q_value': [weighted_average(res_df, 'q_value', weight)],
                             'num': [weighted_average(res_df, 'num', weight)]
                             })
    res_df = pd.concat([res_df, final_df], ignore_index=True)
    decimal_places = {
                      'q_value': 2,
                     }
    for col, decimals in decimal_places.items():
        res_df[col] = res_df[col].round(decimals)
    res_df.to_csv(os.path.join(eval_root, 'q_sum.csv'), index=False, )