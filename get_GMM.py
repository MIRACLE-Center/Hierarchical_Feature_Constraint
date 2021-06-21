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
import pickle

from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import sklearn.covariance
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from copy import deepcopy


def run(args, default_victims):
    # root_dir = f'/apdcephfs/share_1290796/qingsongyao/temp/{args.dataset}/{args.arch}/new_GMM_{args.num_component}'
    root_dir = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch, f'GMM_{args.num_component}')
    if not os.path.isdir(root_dir): os.mkdir(root_dir)
    layer_index = args.layer_index
    is_targeted = False
    saver = Saver(args.arch, dataset=args.dataset)
    logging.info(f'Attacking {default_victims} by {args.arch}')

    test_loader = get_dataloader(dataset=args.dataset, mode='train',\
        batch_size=args.batch_size, num_workers=args.workers, \
            num_fold=args.num_fold, targeted=is_targeted)

    if args.arch == 'vgg16':
        src_model = infer_Cls_Net_vgg(test_loader.dataset.num_classes)
    elif args.arch == 'resnet50':
        src_model = infer_Cls_Net_resnet(test_loader.dataset.num_classes)
    src_model = saver.load_model(src_model, args.arch)
    src_model.eval()
    src_model = src_model.cuda()


    def fit(id_class, id_layer, features):
        print(f'Start to fit {id_class} on layer {id_layer}')
        gmm_model = BayesianGaussianMixture(n_components=args.num_component, n_init=2, max_iter=1000)
        gmm_model.fit(features)
        npz_name = os.path.join(root_dir, f'Layer_{id_layer}_class_{id_class}.npz')
        np.savez(npz_name, means_=gmm_model.means_,\
            precision_cholesky_=gmm_model.precisions_cholesky_,\
            weights_=gmm_model.weights_)
        print(f'Fit {id_class} on layer {id_layer} done save as {npz_name}')

    from multiprocessing import Process
    for id_class in range(0, test_loader.dataset.num_classes_selected):
        features_layer = {id:[] for id in range(0, src_model.num_cnn)}
        test_loader = get_dataloader(dataset=args.dataset, mode='train',\
            batch_size=32, num_workers=args.workers, \
            rand_pairs='train_single_class', target_class=id_class)
        feature_list = list()
        # for i, (images, target) in enumerate(test_loader):
        for repeat in range(1):
            for i, (images, target, target_data) in enumerate(tqdm(test_loader)):
                images = images.cuda()
                with torch.no_grad():
                    target_feature = src_model.feature_list(images)[1]
                    for id in range(0, src_model.num_cnn):
                        features_layer[id].append(target_feature[id].mean(-1).mean(-1).detach().cpu().numpy())
        for id in range(0, src_model.num_cnn):
            features_layer[id] = np.concatenate(features_layer[id])

        for id_layer in range(0, src_model.num_cnn):
            fit(id_class, id_layer, features_layer[id_layer])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-n', '--run_name', default='vgg16')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
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
    parser.add_argument('-y', '--num_component', default=2, type=int,
                        help='Fold Number')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    default_victims = {
                       'vgg16_bn':'vgg16_bn', 
                    #    'googlenet':'googlenet',
                    #    'inception_v3':'inception_v3',
                    #    'resnet50':'resnet50',
                    }

    # print(f"Run Attacking {args.run_name} on Jizhi against ")
    run(args, default_victims)
