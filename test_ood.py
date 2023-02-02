import argparse
import os
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.autograd import Variable


from datasets import get_dataloader
from utils import *
from network import *
from saver import Saver
from sklearn.metrics import accuracy_score, roc_auc_score

import random
import PIL

def add_rand_mask(images, length=50):
    rand_x, rand_y = random.randint(70, 180), random.randint(70, 180)
    images[:, :, rand_y:rand_y+length, rand_x:rand_x+length] = 0
    return images

def to_PIL(tensor):
    tensor = tensor.cpu().detach().numpy().transpose(1,2, 0)
    return PIL.Image.fromarray((tensor*255.0).astype(np.uint8))

def run(args, length=10):
    print(f'Mask Length: {length}')
    is_targeted = False
    saver = Saver(args.arch, dataset=args.dataset)

    test_loader = get_dataloader(args.dataset, arch=args.arch, mode='test',\
        batch_size=args.batch_size, num_workers=4, \
        num_fold=args.num_fold, targeted=is_targeted)
    num_classes = test_loader.dataset.num_classes

    if 'vgg16' in args.arch:
        src_model = infer_Cls_Net_vgg(num_classes)
    elif 'resnet50' in args.arch:
        src_model = infer_Cls_Net_resnet(num_classes)
    elif 'resnet3d' in args.arch:
        src_model = infer_Cls_Net_resnet3d(num_classes)
    else:
        raise NotImplementedError

    src_model = saver.load_model(src_model, args.arch)
    src_model.eval()
    src_model = src_model.cuda()

    num_layers = src_model.num_feature
    num_cnn_layers = src_model.num_cnn

    def loss_fn_(feature, item):
        

        final_loss = []
        mean_list, std_list, weights = item[1]

        # Compute HFC loss
        for id_layer in range(num_cnn_layers):
            # if id_layer < num_cnn_layers - 1: continue
            temp_mean = mean_list[id_layer]
            std = std_list[id_layer]
            weight = weights[id_layer].log()

            # Skip the layers with too large weights with big HFC losses
            shape = feature[id_layer].shape

            # Compute mean and prob to n_conponent
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
                # Find the nearest component
                selected_comp = scores.argmax(-1)
            
            zero_f = mean_feature - temp_mean[selected_comp]
            tmp =  torch.bmm(zero_f.unsqueeze(1), std[selected_comp]).squeeze()
            hfc_loss = (tmp * tmp).mean(-1).mean().squeeze()
            final_loss.append(hfc_loss)
        return  torch.stack(final_loss).cpu().detach().numpy()

    num_component_gmm = args.num_component
    root_gmm = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch, \
        f'GMM_{num_component_gmm}')
    print(f'load GMM from {root_gmm}')
    
    total_clean_scores = []
    total_ood_scores = []
    for id_class in range(2):
        test_loader = get_dataloader(dataset=args.dataset, arch=args.arch, mode='adv_test',\
            batch_size=args.batch_size, num_workers=4, \
            num_fold=args.num_fold, targeted=is_targeted, rand_pairs='specific', target_class=id_class)

        # Load GMM models
        mean_list, pre_chol_list, weight_list, scores = [], [], [], []
        for id_layer in range(num_cnn_layers):
            gmm_model = np.load(os.path.join(root_gmm, f'Layer_{id_layer}_class_{1 - id_class}.npz'))
            mean_list.append(torch.from_numpy(gmm_model['means_']).cuda().float())
            pre_chol_list.append(torch.from_numpy(gmm_model['precision_cholesky_']).cuda().float())
            weight_list.append(torch.from_numpy(gmm_model['weights_']).cuda().float())

        for i, (images, target) in enumerate(tqdm(test_loader, desc=f'Class {1 - id_class}')):
            if True:
                images = images.cuda()
                target = target.cuda()
                # false_images = false_images.cuda()

            bingo_clean = src_model(images).argmax(dim=1).detach().cpu().item() == 1 - target
            assert(bingo_clean == 1)

            # to_PIL(images[0]).save('clean.png')
            
            hfc_scores = loss_fn_(src_model.feature_list(images)[1], [target, [mean_list, pre_chol_list, weight_list]])
            
            ood_images = add_rand_mask(images, length=length)
            # to_PIL(ood_images[0]).save('OOD.png')
            bingo_ood = src_model(ood_images).argmax(dim=1).detach().cpu().item() == 1 - target
            if not bingo_ood:
                print('False predicted !')
                continue
            ood_hfc_scores = loss_fn_(src_model.feature_list(ood_images)[1], [target, [mean_list, pre_chol_list, weight_list]])
            total_ood_scores.append(ood_hfc_scores)
            total_clean_scores.append(hfc_scores)
    
    total_ood_scores = np.stack(total_ood_scores)
    total_clean_scores = np.stack(total_clean_scores)
    number = total_ood_scores.shape[0]
    gt = np.stack([np.zeros([number]), np.ones([number])])
    total_clean_scores = total_clean_scores.transpose()
    total_ood_scores = total_ood_scores.transpose()

    for i in range(47, 48):
        pred = np.stack([total_clean_scores[i], total_ood_scores[i]])
        auc = roc_auc_score(gt, pred)
        print(f'Layer {i+1} AUC {auc}')

    return auc

    # calculate metrics
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training Code')
    parser.add_argument('--root_dir', metavar='DIR', default='/apdcephfs/share_1290796/qingsongyao/SecureMedIA/dataset/aptos2019/',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-f', '--num_fold', default=0, type=int,
                        help='Fold Number')
    parser.add_argument('-y', '--num_component', default=1, type=int,
                        help='num_component')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    parser.add_argument('--attack', default='I_FGSM_Linf_1', type=str,
                        help='Type of the adversarial attack') 
    parser.add_argument('--detector', default='I_FGSM_Linf_1', type=str,
                        help='Choose detector trained by which adversarial attack')  
    parser.add_argument('--lamda', default=1, type=float,
                        help='lamda') 
    parser.add_argument('--get_feature', default=0, type=float,
                        help='get feature layer index')                   
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    length_setting = [1, 2, 4, 8, 16, 32, 64]
    auc_list = []
    for i in length_setting:
        det_auc = run(args, i)
        auc_list.append(det_auc)
    auc_list = np.array(auc_list)
    length_setting = np.array(length_setting)

    # rewrite = True
    # save_path = f'temp.pkl'
    # if not os.path.isfile(save_path) or rewrite:
    #     with open(save_path, 'wb') as f:
    #         pickle.dump([length_setting, auc_list], f)
    # else:
    #     with open(save_path, 'rb') as f:
    #         ength_setting, auc_list = pickle.load(f)

    

