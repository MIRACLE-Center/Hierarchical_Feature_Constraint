import torch
import torch.nn as nn
import time
import logging
import numpy as np

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

class my_sequential(nn.Module):
    def __init__(self, *layers):
        super(my_sequential, self).__init__()
        self.num_feature = 0
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)
            if type(layer) == torch.nn.ReLU:
                self.num_feature += 1
        self.num_layers = i + 1

    def forward(self, x, get_features=False, return_index=None):
        feature = []
        for i in range(self.num_layers):
            x = getattr(self, str(i))(x)
            if get_features and type(getattr(self, str(i))) == torch.nn.ReLU:
                feature.append(x)
                if len(feature) - 1 == return_index: return feature[-1]
        if get_features: return x, feature
        return x

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def gen_results_strings(label, results_raw, results_adv):
    out_string = f"Label {label} \n"
    raw_str = "Before Attack : \n"
    for i in range(results_raw.shape[0]):
        raw_str += f"Class: {i} Prob: {results_raw[i]} \n"
    adv_str = "After Attack : \n"
    for i in range(results_adv.shape[0]):
        adv_str += f"Class: {i} Prob: {results_adv[i]} \n"
    return out_string + raw_str + adv_str

def get_TPR_at_TNR(X_data, num_normal, rate=10):
    normal_data = X_data[:num_normal]
    adv_data = X_data[num_normal:]
    threshold = np.percentile(normal_data, rate, 0)
    return 1 - (adv_data > threshold).sum() / adv_data.shape[0]

from tqdm import tqdm
def check_adv_samples(data_dict, model, label):
    model.eval()
    for key, data in data_dict.items():
        batch_size = 100
        total = 0
        bingo = 0
        for data_index in range(int(np.floor(data.shape[0]/batch_size))):
            data = torch.from_numpy(data_dict[key][total : total + batch_size]).cuda().float()
            target = torch.from_numpy(label[total : total + batch_size]).cuda()
            total += batch_size
            output = model.feature_list(data)[0].argmax(1)
            bingo += (target == output).sum().item()
        print(f'Check data: {key} ACC {float(bingo / total)}')

def get_pairs_auc(normal, noise, adv, split=0.3, adv_test=False, negative=True, get_rate=False):
    # Gen train test pairs and compute auc
    # Input [num_data] output auc score
    num_data = normal.shape[0]
    num_test = int(num_data * split)
    if not adv_test:
        X_data = np.concatenate([normal[-num_test:], noise[-num_test:], adv[-num_test:]])
    else:
        X_data = np.concatenate([normal[-num_test:], noise[-num_test:], adv])

    # They are all < 0
    if X_data.mean() > 0 and negative: X_data = -X_data
    # Z_score
    # X_data = scale(X_data)

    if not adv_test:
        Y_data = np.concatenate([np.ones_like(normal[-num_test:]), \
            np.ones_like(normal[-num_test:]), np.zeros_like(normal[-num_test:])])
    else:
        Y_data = np.concatenate([np.ones_like(normal[-num_test:]), \
            np.ones_like(normal[-num_test:]), np.zeros_like(adv)])
    # if normal.mean() < adv.mean():
    #     Y_data = 1 - Y_data
    aucroc = roc_auc_score(Y_data, X_data)
    if get_rate: return aucroc, get_TPR_at_TNR(X_data, 2*num_test)
    return aucroc

def logits_regression_auc(normal, noise, adv, split=0.3):
    # Input [num_data, num_features] 
    # MinMaxScale
    # Train logits regression and eval
    # finally output auc score
    num_data = normal.shape[0]

    X_total = np.concatenate([normal, noise, adv])
    scaler = StandardScaler().fit(X_total)
    X_total = scaler.transform(X_total) 

    normal = X_total[:num_data]
    noise = X_total[num_data:2*num_data]
    adv = X_total[2*num_data:]

    num_data = normal.shape[0]
    num_test = int(num_data * split)
    X_train = np.concatenate([normal[:-num_test], noise[:-num_test], adv[:-num_test]])
    Y_train = np.concatenate([np.ones(normal[:-num_test].shape[0]), \
        np.ones(normal[:-num_test].shape[0]), np.zeros(normal[:-num_test].shape[0])])

    lr = LogisticRegression(n_jobs=-1, max_iter=500, solver='lbfgs').fit(X_train, Y_train)
    X_test = np.concatenate([normal[-num_test:], noise[-num_test:], adv[-num_test:]])
    Y_test = np.concatenate([np.ones(normal[-num_test:].shape[0]), \
        np.ones(normal[-num_test:].shape[0]), np.zeros(normal[-num_test:].shape[0])])
    pred_prob = lr.predict_proba(X_test).transpose()[1]
    return roc_auc_score(Y_test, pred_prob), [lr, scaler]
    
def logits_regression_infer(normal, noise, adv, lr, split=0.3, get_rate=False):
    num_data = normal.shape[0]
    lr, scaler = lr
    num_test = int(num_data * split)

    X_total = np.concatenate([normal[-num_test:], noise[-num_test:], adv])
    X_total = scaler.transform(X_total) 
    
    Y_test = np.concatenate([np.ones(normal[-num_test:].shape[0]), \
        np.ones(normal[-num_test:].shape[0]), np.zeros(adv.shape[0])])
    pred_prob = lr.predict_proba(X_total).transpose()[1]

    aucroc = roc_auc_score(Y_test, pred_prob)
    if get_rate: return aucroc, get_TPR_at_TNR(pred_prob, 2*num_test)
    return aucroc

def logits_regression_infer_raw(normal, noise, adv, lr, split=0.3, get_rate=False):
    num_data = normal.shape[0]
    lr, scaler = lr
    num_test = int(num_data * split)

    X_total = np.concatenate([normal[-num_test:], noise[-num_test:], adv[-num_test:]])
    X_total = scaler.transform(X_total) 
    
    Y_test = np.concatenate([np.ones(normal[-num_test:].shape[0]), \
        np.ones(normal[-num_test:].shape[0]), np.zeros(adv[-num_test:].shape[0])])
    pred_prob = lr.predict_proba(X_total).transpose()[1]

    aucroc = roc_auc_score(Y_test, pred_prob)
    if get_rate: return aucroc, get_TPR_at_TNR(pred_prob, 2*num_test)
    return aucroc
