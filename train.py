import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from torchvision.models.inception import InceptionOutputs
from torchvision.models.googlenet import GoogLeNetOutputs

import my_models as models

from datasets import get_dataloader
from utils import *
from network import Cls_Net
from saver import Saver

from sklearn.metrics import accuracy_score

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def train(train_loader, model, criterion, optimizer, epoch, args):
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')

    model.train()

    for images, target in tqdm(train_loader, desc=f"Train epoch {epoch}"):
        if args.gpu:
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        if type(output) == InceptionOutputs: output = output[0]
        if type(output) == GoogLeNetOutputs: output = output[0]
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Epoch [{}][{}] Acc@1 {top1.avg:.3f} Loss {loss.avg:.3f}, Lr {lr:.9f}'
            .format(epoch, len(train_loader), top1=top1, \
                loss=losses, lr=optimizer.param_groups[0]['lr']))

def eval(test_loader, model, criterion, optimizer, epoch, args, saver=None):
    losses = AverageMeter('Loss Eval', ':.4e')
    progress = ProgressMeter(
        len(test_loader),
        [losses],
        prefix="Epoch Eval: [{}]".format(epoch))

    model.eval()

    gt = np.array([])
    pred = np.array([])
    logits = np.array([])
    for i, (images, target) in enumerate(tqdm(test_loader, desc=f"Test epoch {epoch}")):
        if args.gpu:
            images = images.cuda()
            target = target.cuda()

        # compute output
        with torch.no_grad():
            output = model(images)
        if type(output) == InceptionOutputs: output = output[0]
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))

        # Record GT and Pred
        gt = np.concatenate([gt, target.detach().cpu().numpy()], axis=0)
        output_argmax = output.argmax(dim=1).detach().cpu().numpy()
        pred = np.concatenate([pred, output_argmax], axis=0)

        output = torch.nn.functional.softmax(output, dim=1)
        logits = np.concatenate([logits, output.transpose(1,0)[1].detach().cpu().numpy()], axis=0)
    
    acc = accuracy_score(gt, pred, normalize=True)

    print('Eval Epoch [{}][{}] Acc {acc:.5f} loss {loss.avg:.3f}'
            .format(epoch, len(test_loader), acc=acc, loss=losses))
    
    return losses.avg, acc

def run(args):
    saver = Saver(args.run_name, dataset=args.dataset)
    print(f"********** Start training {args.run_name} backbone {args.arch} dataset {args.dataset} *******")
    # # test models
    # for item in ['vgg16_bn','resnet101','alexnet','inception_v3']:
    #     model = Cls_Net(5, item)

    train_loader = get_dataloader(dataset=args.dataset, mode='train',\
        batch_size=args.batch_size, num_workers=args.workers, num_fold=args.num_fold)
    test_loader = get_dataloader(dataset=args.dataset, mode='test',\
        batch_size=args.batch_size, num_workers=args.workers, num_fold=args.num_fold)

    model = Cls_Net(train_loader.dataset.num_classes, args.arch)
    if args.gpu: model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    if args.gpu: criterion = criterion.cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    cudnn.benchmark = True
    
    best_loss = 100000
    for epoch in range(args.start_epoch, args.epochs):
        if optimizer.param_groups[0]['lr'] < 1e-7:
            break

        train(train_loader, model, criterion, optimizer, epoch, args)
        
        if epoch % 1 == 0:
            val_loss, val_acc = eval(test_loader, model, criterion, optimizer, epoch, args)
            scheduler.step(val_loss)
            saver.save_current_model(model)
            # import ipdb; ipdb.set_trace()
            # saver.save_best_model(model)

            if best_loss > val_loss:
                best_loss = val_loss
                saver.save_best_model(model)
    eval(test_loader, model, criterion, optimizer, epoch, args, saver)
    print("Training Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('-n', '--run_name', default='test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=8, type=int,
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
    parser.add_argument('--eval', default=None, action='store_true',
                        help='Use GPU or not')
    parser.add_argument('-f', '--num_fold', default=0, type=int,
                        help='Fold Number')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    args = parser.parse_args()
    args.gpu = True

    print(f"Run Training {args.run_name} on Jizhi")
    run(args)
    
