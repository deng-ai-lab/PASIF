from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import torch

METRIC_DICT = {}

def register_transform(name):
    def decorator(cls):
        METRIC_DICT[name] = cls
        return cls
    return decorator

class Evaluator():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.evaluators = {}

        for eval_cfg in cfg:
            self.evaluators[eval_cfg.name + '_' + eval_cfg.get('tag', 'atom')] = METRIC_DICT[eval_cfg.name](**eval_cfg)
    
    def __call__(self, results):
        report = {}
        for name, eval_func in self.evaluators.items():
            report[name] = eval_func(results)
        return report

def merge_list_of_dict(results):
    result_merged = {k:[] for k in results[0].keys()}
    for result in results:
        for k, v in result.items():
            result_merged[k].append(v)
    return {k: torch.cat(v, dim=0) for k, v in result_merged.items()}   

@register_transform('auroc')
class AUROC():
    def __init__(self, true_key, pred_key, mask_key=None, **kwargs) -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.mask_key = mask_key
    
    def __call__(self, results):
        if type(results) is list:
            result_merged = merge_list_of_dict(results)
            auroc_mean = self.cal_auroc(result_merged)
        elif type(results) is dict:
            auroc_mean = self.cal_auroc(results)

        return auroc_mean

    def cal_auroc(self, results):
        y_true = results[self.true_key]
        y_pred = results[self.pred_key]
        if self.mask_key is not None:
            mask = results.get(self.mask_key, torch.ones_like(y_true, dtype=torch.bool))
        else:
            mask = torch.ones_like(y_true, dtype=torch.bool)
        
        y_true = y_true[mask].cpu().numpy().astype(np.int64)
        y_pred = y_pred[mask].cpu().numpy()

        avg_auroc = 0.
        possible_classes = set(y_true)
        for c in possible_classes:
            try:
                auroc = roc_auc_score(y_true == c, y_pred[:, c])
                avg_auroc += auroc * np.sum(y_true == c)
            except:
                auroc = 0.
                avg_auroc += auroc * np.sum(y_true == c)

        return np.divide(avg_auroc, len(y_true))


@register_transform('pearson')
class Pearsonr():
    def __init__(self, true_key, pred_key, mask_key=None, **kwargs) -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.mask_key = mask_key
    
    def __call__(self, results):
        if type(results) is list:
            result_merged = merge_list_of_dict(results)
            auroc_mean = self.cal_pearson(result_merged)
        elif type(results) is dict:
            auroc_mean = self.cal_pearson(results)

        return auroc_mean


    def cal_pearson(self, results):
        y_true = results[self.true_key]
        y_pred = results[self.pred_key]
        if self.mask_key is not None:
            mask = results.get(self.mask_key, torch.ones_like(y_true, dtype=torch.bool))
        else:
            mask = torch.ones_like(y_true, dtype=torch.bool)
        
        y_true = y_true[mask].cpu().numpy()
        y_pred = y_pred[mask].cpu().numpy()

        return pearsonr(y_pred, y_true)[0]

@register_transform('spearman')
class Spearman():
    def __init__(self, true_key, pred_key, mask_key=None, **kwargs) -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.mask_key = mask_key
    
    def __call__(self, results):
        if type(results) is list:
            result_merged = merge_list_of_dict(results)
            auroc_mean = self.cal_spearman(result_merged)
        elif type(results) is dict:
            auroc_mean = self.cal_spearman(results)

        return auroc_mean

    def cal_spearman(self, results):
        y_true = results[self.true_key]
        y_pred = results[self.pred_key]
        if self.mask_key is not None:
            mask = results.get(self.mask_key, torch.ones_like(y_true, dtype=torch.bool))
        else:
            mask = torch.ones_like(y_true, dtype=torch.bool)
        
        y_true = y_true[mask].cpu().numpy()
        y_pred = y_pred[mask].cpu().numpy()

        return spearmanr(y_pred, y_true)[0]
    

@register_transform('r2')
class CoeffDeter():
    def __init__(self, true_key, pred_key, mask_key=None, **kwargs) -> None:
        self.true_key = true_key
        self.pred_key = pred_key
        self.mask_key = mask_key
    
    def __call__(self, results):
        if type(results) is list:
            result_merged = merge_list_of_dict(results)
            auroc_mean = self.cal_r2(result_merged)
        elif type(results) is dict:
            auroc_mean = self.cal_r2(results)

        return auroc_mean

    def cal_r2(self, results):
        y_true = results[self.true_key]
        y_pred = results[self.pred_key]
        if self.mask_key is not None:
            mask = results.get(self.mask_key, torch.ones_like(y_true, dtype=torch.bool))
        else:
            mask = torch.ones_like(y_true, dtype=torch.bool)
        
        y_true = y_true[mask].cpu().numpy()
        y_pred = y_pred[mask].cpu().numpy()

        r2 = r2_score(y_pred, y_true)
        r2 = min(r2, 1.0)
        r2 = max(r2, 0.0)
        return r2