
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision.models.inception import InceptionOutputs
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets import get_dataloader
from utils import *
from network import infer_Cls_Net_vgg, infer_Cls_Net_resnet
from saver import Saver
import attackers

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from PIL import Image

def to_PIL(image):
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image.transpose(1, 2, 0))

def run(args, default_victims, attack_methods):
    base_dir = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch)
    ID = 0
    if not os.path.isdir(base_dir): os.mkdir(base_dir)
    gt = np.load(f'{base_dir}/gt.npy')
    clean_image = np.load(f'{base_dir}/clean.npy')
    clean_image = clean_image[gt == 0]
    
    to_PIL(clean_image[ID]).save(f'adv_clean.png')
    for item in [1, 2, 4, 8, 16]:

        path = f'{base_dir}/PGD_Linf_{item}.npy'
        adv_image = np.load(path)
        adv_image = adv_image[gt == 0]
        print(np.abs((clean_image - adv_image)).max() * 256)
        to_PIL(adv_image[ID]).save(f'adv_{item}.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
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
    parser.add_argument('--attack', default='I_FGSM_Linf_1', type=str,
                        help='Fold Number')                        
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    default_victims = {
                    #    'vgg16_bn':'vgg16_bn', 
                    #    'googlenet':'googlenet',
                    #    'inception_v3':'inception_v3',
                       'resnet50':'resnet50',
                    }
    print(f"Run Attacking {args.arch}")
    run(args, default_victims, args.attack)