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
from sklearn.mixture import GaussianMixture

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def normalize(tensor):
    tensor = tensor.mul(attackers.torch_std_cuda)\
        .add(attackers.torch_mean_cuda)
    return tensor

def new_model(model):
    def func(input):
        # input : [image, index]
        return model.feature_list(input), model(input)
    # return model.get_feature_attack
    return func

def run(args, attack_methods):
    assert(len(attack_methods) == 1)
    num_component_gmm = args.num_component

    is_targeted = False
    saver = Saver(args.arch, dataset=args.dataset)

    test_loader = get_dataloader(dataset=args.dataset, arch=args.arch, mode='test',\
        batch_size=args.batch_size, num_workers=args.workers, \
        num_fold=args.num_fold, targeted=is_targeted)
    num_classes = test_loader.dataset.num_classes

    if args.arch == 'vgg16':
        src_model = infer_Cls_Net_vgg(num_classes)
    elif args.arch == 'resnet50':
        src_model = infer_Cls_Net_resnet(num_classes)
    else:
        raise NotImplementedError

    src_model = saver.load_model(src_model, args.arch)
    src_model.eval()
    src_model = src_model.cuda()

    num_layers = src_model.num_feature
    num_cnn_layers = src_model.num_cnn

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
        metric_counter[attacker_name]['mse'] = list()
        metric_counter[attacker_name]['mse_raw'] = list()

    loss_fn = torch.nn.MSELoss()
    bce_loss_fn = torch.nn.CrossEntropyLoss()
    def fn_cw_loss(logits, target, get_scaler=True):
        one_hot = torch.zeros_like(logits)
        one_hot = one_hot.scatter(1, target.view(-1,1), 1)
        target_logits = (one_hot * logits).sum(-1)
        remaining_logits_max = ((1 - one_hot) * logits).max(-1)[0]
        if get_scaler:
            cw_loss = torch.clamp(remaining_logits_max - target_logits + 10, min=0.).mean()
        else:
            cw_loss = torch.clamp(remaining_logits_max - target_logits + 10, min=0.)
        # print('wtf')
        return cw_loss
    if 'CW' in attack_methods[0]:
        bce_loss_fn = fn_cw_loss

    def loss_fn_(feature, item):
        
        feature, logits = feature
        feature = feature[1]
        target = item[0]
        bce_loss = bce_loss_fn(logits, target)
        final_loss = bce_loss
        mean_list, std_list, weights = item[1]

        # temp_mean = mean.index_select(0, target)
        for id_layer in range(num_cnn_layers):
            temp_mean = mean_list[id_layer]
            std = std_list[id_layer]
            weight = weights[id_layer].log()

            # Compute mean and prob to n_conponent
            shape = feature[id_layer].shape
            if shape[1] > 512: continue
            mean_feature = feature[id_layer].view(feature[id_layer].size(0), \
                feature[id_layer].size(1), -1).mean(-1)
            num_component = temp_mean.shape[0]
            
            with torch.no_grad():
                select_featue = mean_feature.unsqueeze(1).repeat((1,num_component,1))
                # mean : [batch, n_comp, n_fea] == zero_f
                zero_f = select_featue - temp_mean
                scores = torch.zeros([zero_f.shape[0], zero_f.shape[1]]).float().cuda()
                for id_comp in range(num_component):
                    id_mean = zero_f[:,id_comp,:]
                    hfc_score = torch.mm(id_mean, std[id_comp]) * torch.mm(id_mean, std[id_comp])
                    score = -0.5 * hfc_score.sum(-1)
                    score = score + std[id_comp].diag().log().sum()
                    scores[:, id_comp] = score
                scores += weight
                selected_comp = scores.argmax(-1)
            
            zero_f = mean_feature - temp_mean[selected_comp]
            tmp =  torch.bmm(zero_f.unsqueeze(1), std[selected_comp]).squeeze()
            # import ipdb; ipdb.set_trace()
            hfc_loss = (tmp * tmp).mean(-1).mean()
            final_loss =  args.lamda * hfc_loss + final_loss
        # print('bce {:.3f} final {:.3f}'.format(bce_loss, final_loss))
        return  final_loss
    loss_fn = loss_fn_
        # import ipdb; ipdb.set_trace()


    # Gen pertubations via attack methods
    attacker_dict = dict()
    for name in attack_methods:
        splits = name.split('_')
        attacker_name = ('_').join(splits[:-1])
        epsilon = 2 * float(name.split('_')[-1]) / 256
        # L2 constrain epsilon * sqrt(c * w * h)
        print(f'Attack by Constrain: {255 * epsilon / 2}')
        if 'L2' in name:
            epsilon = epsilon * np.sqrt(test_loader.dataset.__getitem__(0)[0].view(-1).shape[0])
        elif 'L1' in name:
            epsilon = epsilon * test_loader.dataset.__getitem__(0)[0].view(-1).shape[0]
        else:
            # 'Linf'
            pass
        # if 'CW' in attacker_name:
        #     loss_fn = loss_fn_cw
        kwargs = {
            'predict' : new_model(src_model),
            'loss_fn' : loss_fn,
            'eps' : epsilon,
            'clip_min' : -1,
            'clip_max' : 1,
            'targeted' : True, }

        if args.dataset == 'APTOS' and args.arch == 'vgg16':
            kwargs['eps_iter'] = 2 * 2 * 2 / 256 / 20
        elif args.dataset == 'APTOS' and args.arch == 'resnet50':
            kwargs['eps_iter'] = 2 * 2 * 0.5 / 256 / 20
        elif args.dataset == 'CXR' and args.arch == 'vgg16':
            kwargs['eps_iter'] = 2 * 2 * 2 / 256 / 50
        elif args.dataset == 'CXR' and args.arch == 'resnet50':
            kwargs['eps_iter'] = 2 * 2 * 0.5 / 256 / 50
        elif args.dataset == 'Cifar' and args.arch == 'resnet50':
            kwargs['eps_iter'] = 2 * 2 * 0.5 / 256 / 10
        else:
            raise NotImplementedError

        if attacker_name == 'FGSM_Linf' or attacker_name == 'FGSM_L2':
            kwargs['nb_iter'] = 1
            kwargs['eps_iter'] = epsilon
        elif attacker_name == 'M_FGSM_L2' or attacker_name == 'MI_FGSM_Linf':
            kwargs['nb_iter'] = 2 * int(epsilon / kwargs['eps_iter'])
        # elif attacker_name == 'D_M_FGSM_L2' or attacker_name == 'DI_MI_FGSM_Linf':
        #     kwargs['momentum'] = 1
        #     kwargs['diversity_prob'] = 0.5
        elif attacker_name == 'PGD_Linf' or attacker_name == 'PGD_L2':
            kwargs['rand_init'] = True
            kwargs['nb_iter'] = 2 * int(epsilon / kwargs['eps_iter'])
        elif attacker_name == 'Noise_Linf' or attacker_name == 'Noise_L2':
            kwargs['nb_iter'] = 0
            kwargs['rand_init'] = True
        elif attacker_name == 'CW_L2':
            kwargs['nb_iter'] = 100
            if args.lamda: kwargs['initial_const'] = 0.001
            else: kwargs['initial_const'] = 0.1
        elif attacker_name == 'EAD_L1':
            kwargs['nb_iter'] = 100
            kwargs['const_L1'] = 0.01
        else:
            kwargs['nb_iter'] = 2 * int(epsilon / kwargs['eps_iter'])
            # BIM Default kwargs
            pass
        print('Iteration: {}'.format(kwargs['nb_iter']))

        attacker = attackers.__dict__[attacker_name](\
            **kwargs)
        attacker_dict[name] = attacker

    clean_feature_dict = dict()
    adv_feature_dict = dict()
    for i in range(2):
        clean_feature_dict[i] = list()
        adv_feature_dict[i] = list()

    total_feature_list = {item:[] for item in range(num_layers)}
    total_pred_list = list()
    clean_images = list() 
    root_gmm = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch, \
        f'GMM_{num_component_gmm}')
    print(f'load GMM from {root_gmm}')
    # Start to attack
    for id_class in range(2):
        test_loader = get_dataloader(dataset=args.dataset, arch=args.arch, mode='adv_test',\
            batch_size=args.batch_size, num_workers=args.workers, \
            num_fold=args.num_fold, targeted=is_targeted, rand_pairs='specific', target_class=id_class)

        mean_list, pre_chol_list, weight_list, scores = [], [], [], []
        for id_layer in range(num_cnn_layers):
            gmm_model = np.load(os.path.join(root_gmm, f'Layer_{id_layer}_class_{id_class}.npz'))
            # Here precision is a metric with shape [n_components, n_featues, n_featues]
            # gmm.means_ shape [n_components, n_features]
            # precision[i] = covirence[i].inverse() (ni ju zhen)
            # precision cholesky is the cholesky decomposition of precision (for better computing only)
            mean_list.append(torch.from_numpy(gmm_model['means_']).cuda().float())
            pre_chol_list.append(torch.from_numpy(gmm_model['precision_cholesky_']).cuda().float())
            weight_list.append(torch.from_numpy(gmm_model['weights_']).cuda().float())

        for i, (images, target) in enumerate(tqdm(test_loader, desc=f'Class {id_class}')):
            # if i > 2: break
            if True:
                images = images.cuda()
                clean_images.append(images.cpu().numpy())
                target = target.cuda()
                # false_images = false_images.cuda()
            
            for attacker_name, attacker in attacker_dict.items():
                adv_images = attacker.perturb(images, [target, [mean_list, pre_chol_list, weight_list]])
                # pertubation = (normalize(adv_images) - normalize(images))

                output = src_model(adv_images).argmax(dim=1).detach().cpu().numpy()
                metric_counter[attacker_name]['data'].append(adv_images.cpu().numpy())
                gt_concatnate = target.detach().cpu().numpy()
                metric_counter[attacker_name]['gt'] = \
                    np.concatenate([metric_counter[attacker_name]['gt'], gt_concatnate], axis=0)
                metric_counter[attacker_name]['pred'] = \
                    np.concatenate([metric_counter[attacker_name]['pred'], output], axis=0)
        
                # # Get feature distribution
                # for j in range(images.shape[0]):
                #     real_feature = src_model.get_feature(images, 7)[j].detach()
                #     adv_feature = src_model.get_feature(adv_images, 7)[j].detach()
                #     num_channel = real_feature.shape[0]
                #     real_feature = real_feature.view(num_channel, -1).mean(-1).cpu().unsqueeze(0).numpy()
                #     adv_feature = adv_feature.view(num_channel, -1).mean(-1).cpu().unsqueeze(0).numpy()
                #     clean_feature_dict[target[j].item()].append(real_feature)
                #     adv_feature_dict[output[j]].append(adv_feature)
                
                adv_feature_list = src_model.feature_list(adv_images)[1]
                for id_layer, item in enumerate(adv_feature_list):
                    temp = item.view(item.shape[0], item.shape[1], -1).mean(-1).detach().cpu().numpy()
                    total_feature_list[id_layer].append(temp)
                total_pred_list.append(output)

    # # import ipdb; ipdb.set_trace()    
    # clean_0 = np.array(clean_feature_dict[0]).transpose()
    # np.save(os.path.join(save_root, 'clean_0.npy'), clean_0)
    # clean_1 = np.array(clean_feature_dict[1]).transpose()
    # np.save(os.path.join(save_root, 'clean_1.npy'), clean_1)
    # adv_0 = np.array(adv_feature_dict[0]).transpose()
    # np.save(os.path.join(save_root, 'adv_0.npy'), adv_0)
    # adv_1 = np.array(adv_feature_dict[1]).transpose()
    # np.save(os.path.join(save_root, 'adv_1.npy'), adv_1)


    # call metrics
    for attacker_name, attacker in attacker_dict.items():
        metric_counter[attacker_name]['acc'] = accuracy_score(\
            metric_counter[attacker_name]['gt'], \
            metric_counter[attacker_name]['pred'], normalize=True)
        print("Acc {:.3f} Using {}".format(\
            metric_counter[attacker_name]['acc'], attacker_name))
    
    save_imgs = np.concatenate(metric_counter[attacker_name]['data'], axis=0)
    save_dir = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch, args.attack)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(f'{save_dir}/GMM_{num_component_gmm}.npy', save_imgs)

    from eval import Feature_Detector
    detector = Feature_Detector(temp_dir=\
        os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch, args.detector),\
            num_classes=num_classes, num_layers=num_layers, num_cnn_layers=num_cnn_layers)
    # detector = Feature_Detector(temp_dir=\
    #     f'/apdcephfs/share_1290796/qingsongyao/temp/{args.dataset}/{args.arch}/I_FGSM_Linf_2',\
    #         num_classes=num_classes, num_layers=num_layers, num_cnn_layers=num_cnn_layers)
    total_pred_list = np.concatenate(total_pred_list, axis=0)
    for i in range(num_layers):
        total_feature_list[i] = np.concatenate(total_feature_list[i], axis=0)
    np.save(f'{save_dir}/GMM_{num_component_gmm}_pred.npy', total_pred_list)

    adv_data = torch.from_numpy(save_imgs).cuda().float()
    bingo = metric_counter[attacker_name]['gt'] == metric_counter[attacker_name]['pred']
    correct_image = adv_data[bingo]
    labels = metric_counter[attacker_name]['gt'][bingo]
    if labels.shape[0] % 100 == 1:
        correct_image = correct_image[:100 * (labels.shape[0]/100)]
        labels = labels[:100 * (labels.shape[0]/100)]
    detector.eval_patch(correct_image, src_model, labels)

    clean_images = np.concatenate(clean_images, axis=0)
    diff = np.abs(clean_images - save_imgs) * 255 / 2
    diff = diff.reshape(diff.shape[0], -1)
    num_dims = diff.shape[-1]
    print(f'L inf max {diff.max()}')
    print(f'L 2 {(np.sqrt((diff * diff).sum(-1) / num_dims)).mean()}')
    print(f'L 1 {(diff.sum(-1) / num_dims).mean()}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-n', '--run_name', default='vgg16')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16')
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
    parser.add_argument('-i', '--num_layers', default=15, type=int,
                        help='Fold Number')
    parser.add_argument('-u', '--num_cnn_layers', default=13, type=int,
                        help='Fold Number')
    parser.add_argument('-y', '--num_component', default=1, type=int,
                        help='Fold Number')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    parser.add_argument('--attack', default='I_FGSM_Linf_1', type=str,
                        help='Fold Number') 
    parser.add_argument('--detector', default='I_FGSM_Linf_1', type=str,
                        help='Fold Number')  
    parser.add_argument('--lamda', default=1, type=float,
                        help='Fold Number')                      
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    print(f"Run Attacking {args.arch} on Jizhi against ")
    run(args, [args.attack])
