import argparse
import os
import random
import shutil
import time
import PIL
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from io import BytesIO
import numpy as np
import math

from datasets import get_dataloader
from utils import *
from network import infer_Cls_Net_vgg, infer_Cls_Net_resnet
from saver import Saver
import attackers

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def defend_reduce(arr, depth=1):
    arr = arr.cpu().detach().numpy()
    arr = (arr * 255.0).astype(np.uint8)
    shift = 8 - depth
    arr = (arr >> shift) << shift
    arr = arr.astype(np.float32)/255.0
    return torch.from_numpy(arr).cuda()

def defend_jpeg(input_array):
    input_array = input_array.cpu().detach().numpy()
    pil_image = PIL.Image.fromarray((input_array*255.0).astype(np.uint8))
    f = BytesIO()
    pil_image.save(f, format='jpeg', quality=50) # quality level specified in paper
    jpeg_image = np.asarray(PIL.Image.open(f)).astype(np.float32)/255.0
    return torch.from_numpy(jpeg_image).cuda()


def normalize(tensor):
    tensor = tensor.mul(attackers.torch_std_cuda)\
        .add(attackers.torch_mean_cuda)
    return tensor

def new_model(model):
    def func(input):
        return model(input)
    # return model.get_feature_attack
    return func

def adv_attack_run(loss_fn, model, dataloader, attack_methods):

    # Great Metric Savers
    metric_counter = dict()
    for attacker_name in attack_methods:
        metric_counter[attacker_name] = dict()
        gt = np.array([])
        pred = np.array([])
        metric_counter[attacker_name] = dict()
        metric_counter[attacker_name]['gt'] = gt
        metric_counter[attacker_name]['pred'] = pred
        metric_counter[attacker_name]['data'] = list()


    # Gen pertubations via attack methods
    attacker_dict = dict()
    for name in attack_methods:
        splits = name.split('_')
        attacker_name = ('_').join(splits[:-1])
        epsilon = float(name.split('_')[-1]) / 255
        # L2 constrain epsilon * sqrt(c * w * h)
        # print(f'Attack by Constrain: {255 * epsilon}')
        if 'L2' in name:
            epsilon = epsilon * np.sqrt(dataloader.dataset.__getitem__(0)[0].view(-1).shape[0])
        elif 'L1' in name:
            epsilon = epsilon * dataloader.dataset.__getitem__(0)[0].view(-1).shape[0]
        else:
            # 'Linf'
            pass

        kwargs = {
            'predict' : model,
            'loss_fn' : loss_fn,
            'eps' : epsilon,
            'clip_min' : 0,
            'clip_max' : 1,
            'targeted' : False, }

        kwargs['eps_iter'] = 2 / 5 / 255
        if attacker_name == 'FGSM_Linf' or attacker_name == 'FGSM_L2':
            kwargs['nb_iter'] = 1
        elif attacker_name == 'M_FGSM_L2' or attacker_name == 'MI_FGSM_Linf':
            kwargs['nb_iter'] = math.ceil(2 * epsilon / kwargs['eps_iter'])
        # elif attacker_name == 'D_M_FGSM_L2' or attacker_name == 'DI_MI_FGSM_Linf':
        #     kwargs['momentum'] = 1
        #     kwargs['diversity_prob'] = 0.5
        elif attacker_name == 'PGD_Linf' or attacker_name == 'PGD_L2':
            kwargs['rand_init'] = True
            kwargs['nb_iter'] = math.ceil(2 * epsilon / kwargs['eps_iter'])
        elif attacker_name == 'Noise_Linf' or attacker_name == 'Noise_L2':
            kwargs['nb_iter'] = 0
            kwargs['rand_init'] = True
        elif attacker_name == 'CW_L2':
            kwargs['nb_iter'] = 100
        elif attacker_name == 'EAD_L1':
            kwargs['nb_iter'] = 30
            kwargs['const_L1'] = 0.01
        else:
            kwargs['nb_iter'] = math.ceil(2 * epsilon / kwargs['eps_iter'])
            # BIM Default kwargs
            pass
        attacker = attackers.__dict__[attacker_name](\
            **kwargs)
        attacker_dict[name] = attacker
 
    total_labels_list = list()
    loss_mse = torch.nn.MSELoss(reduction='none')
    # Start to attack

    clean_images = list()
    for i, (images, target) in enumerate(dataloader):
        # if i > 2: break
        clean_images.append(images.cpu().numpy())
        if True:
            images = images.cuda()
            target = target.cuda()
            # false_images = false_images.cuda()
        
        for attacker_name, attacker in attacker_dict.items():
            adv_images = attacker.perturb(images)
            # import ipdb; ipdb.set_trace()
            # pertubation = (normalize(adv_images) - normalize(images))

            output = model(adv_images).argmax(dim=1).detach().cpu().numpy()
            metric_counter[attacker_name]['data'].append(adv_images.cpu().numpy())
            gt_concatnate = target.detach().cpu().numpy()
            metric_counter[attacker_name]['gt'] = \
                np.concatenate([metric_counter[attacker_name]['gt'], gt_concatnate], axis=0)
            metric_counter[attacker_name]['pred'] = \
                np.concatenate([metric_counter[attacker_name]['pred'], output], axis=0)
            total_labels_list.append(gt_concatnate)

    # call metrics
    for attacker_name, attacker in attacker_dict.items():
        metric_counter[attacker_name]['acc'] = accuracy_score(\
            metric_counter[attacker_name]['gt'], \
            metric_counter[attacker_name]['pred'], normalize=True)
        print("Adv Acc {:.3f} Using {}".format(\
            metric_counter[attacker_name]['acc'], attacker_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-n', '--run_name', default='resnet50')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=370, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, action='store_true',
                        help='Use GPU or not')
    parser.add_argument('--debug', default=None, action='store_true',
                        help='Use GPU or not')
    parser.add_argument('-f', '--num_fold', default=0, type=int,
                        help='Fold Number')
    parser.add_argument('-i', '--layer_index', default=7, type=int,
                        help='Fold Number')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    parser.add_argument('--attack', default='I_FGSM_Linf_8', type=str, nargs='+',
                        help='Fold Number')                        
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    print(f"Run Attacking {args.arch} on Jizhi against ")

    saver = Saver(args.run_name, dataset=args.dataset)
    print(f"********** Start training {args.run_name} backbone {args.arch} dataset {args.dataset} *******")
    # # test models
    # for item in ['vgg16_bn','resnet101','alexnet','inception_v3']:
    #     model = Cls_Net(5, item)

    train_loader = get_dataloader(dataset=args.dataset, mode='train',\
        batch_size=args.batch_size, num_workers=args.workers, num_fold=args.num_fold)
    test_loader = get_dataloader(dataset=args.dataset, mode='test',\
        batch_size=args.batch_size, num_workers=args.workers, num_fold=args.num_fold)

    tag_cls = 'resnet50'    
    cls_model = infer_Cls_Net_resnet(2)
    cls_saver = Saver(tag_cls, dataset=args.dataset)
    # import ipdb; ipdb.set_trace()
    cls_model = cls_saver.load_model(cls_model, tag_cls)
    model = infer_Cls_Net_resnet()
    if args.gpu: cls_model = cls_model.cuda()
    if args.gpu: model = model.cuda()
    cls_model.eval()

    criterion = nn.CrossEntropyLoss()
    adv_attack_run(criterion, cls_model, test_loader, [args.attack])
