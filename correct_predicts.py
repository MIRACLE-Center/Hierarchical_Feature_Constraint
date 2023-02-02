import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from datasets import get_dataloader
from utils import *
from network import *
from saver import Saver


def run(args):
    saver = Saver(args.arch, dataset=args.dataset)

    test_loader = get_dataloader(dataset=args.dataset, mode='test',\
        batch_size=args.batch_size, num_workers=args.workers, \
        num_fold=args.num_fold, targeted=True, arch=args.arch)

    if args.arch == 'vgg16':
        src_model = infer_Cls_Net_vgg(test_loader.dataset.num_classes)
    elif args.arch == 'resnet50':
        src_model = infer_Cls_Net_resnet(test_loader.dataset.num_classes)
    elif args.arch == 'resnet3d':
        src_model = infer_Cls_Net_resnet3d(test_loader.dataset.num_classes)
    else:
        raise NotImplementedError

    src_model = saver.load_model(src_model, args.arch)
    src_model.eval()
    src_model = src_model.cuda()

    bingo = list()
    diff_logits = list()
    pred_list = list()
    target_list = list()
    for i, (images, target) in enumerate(tqdm(test_loader)):
        images = images.cuda()
        logits = src_model(images).detach()
        pred_value = torch.softmax(logits, dim=-1)[:, 1]
        target_list.append(target.numpy())
        pred_list.append(pred_value.cpu().numpy())
        pred = logits.argmax(1).cpu().detach()
        diff = torch.abs(logits[:,0] - logits[:,1]).cpu() * (pred == target)
        diff_logits.append(diff.numpy())
        bingo.append((pred == target).numpy())
    bingo = np.concatenate(bingo)
    np.save(f'runs_{args.dataset}/{args.arch}/correct_predicts.npy', bingo)
    diff = np.concatenate(diff_logits)
    pred_list = np.concatenate(pred_list)
    target_list = np.concatenate(target_list)
    # auc = roc_auc_score(target_list, pred_list)
    print(f'Mean logits on dataset {args.dataset} by network {args.arch} : {diff.sum()/bingo.sum()} ACC : {bingo.sum() / bingo.shape[0]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
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
    parser.add_argument('--dataset', default='CXR', type=str,
                        help='Fold Number')
    parser.add_argument('--attack', default='I_FGSM_Linf_4', type=str,
                        help='Fold Number')                        
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    run(args)

