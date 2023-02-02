import torch
import torch.nn as nn
import time
import logging
import numpy as np
from scipy import stats as st

from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import torch.nn.functional as F
def input_diversity(x, prob=0):
    img_size = x.shape[-1]
    img_resize = int(img_size * 0.9)

    if 0.9 < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded if torch.rand(1) < prob else x

def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    dim = x.nelement() / batch_size
    # import ipdb; ipdb.set_trace()
    return (x.abs().pow(p).view(batch_size, -1).sum(dim=1) / dim).pow(1. / p)

def clamp_by_pnorm(x, p, r):
    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    if isinstance(r, torch.Tensor):
        assert norm.size() == r.size()
    else:
        assert isinstance(r, float)
    factor = torch.min(r / norm, torch.ones_like(norm))
    return batch_multiply(factor, x)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils

    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)

def l2ball_proj(center, radius, t):
    noise = center - t
    # import ipdb; ipdb.set_trace()
    noise = clamp_by_pnorm(noise, p=2, r=radius)
    return center - noise

def linfball_proj(center, radius, t):
    noise = center - t
    noise = torch.clamp(noise, min=-radius, max=radius)
    return center - noise

def kernel_generation():
    kernel = gkern().astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 1)
    return stack_kernel

def gkern(kernlen=7, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

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

def check_adv_samples(data_dict, model, label):
    model.eval()
    for key, data in data_dict.items():
        batch_size = 10
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
    num_test = int(num_data * split)

    X_total = np.concatenate([normal[-num_test:], noise[-num_test:], adv])
    if lr is not None:
        lr, scaler = lr
    else:
        scaler = StandardScaler().fit(X_total)
    X_total = scaler.transform(X_total)

    Y_test = np.concatenate([np.ones(normal[-num_test:].shape[0]), \
        np.ones(normal[-num_test:].shape[0]), np.zeros(adv.shape[0])])
    if lr is not None:
        pred_prob = lr.predict_proba(X_total).transpose()[1]
    else:
        pred_prob = X_total.mean(-1)

    aucroc = roc_auc_score(Y_test, pred_prob)
    if get_rate: return aucroc, get_TPR_at_TNR(pred_prob, 2*num_test)
    return aucroc

def logits_regression_infer_raw(normal, noise, adv, lr, split=0.3, get_rate=False):
    num_data = normal.shape[0]
    num_test = int(num_data * split)

    X_total = np.concatenate([normal[-num_test:], noise[-num_test:], adv[-num_test:]])
    if lr is not None:
        lr, scaler = lr
        X_total = scaler.transform(X_total) 
    
    Y_test = np.concatenate([np.ones(normal[-num_test:].shape[0]), \
        np.ones(normal[-num_test:].shape[0]), np.zeros(adv[-num_test:].shape[0])])
    if lr is not None:
        pred_prob = lr.predict_proba(X_total).transpose()[1]
    else:
        pred_prob = X_total.mean(-1)
    # import ipdb; ipdb.set_trace()

    aucroc = roc_auc_score(Y_test, pred_prob)
    if get_rate: return aucroc, get_TPR_at_TNR(pred_prob, 2*num_test)
    return aucroc
