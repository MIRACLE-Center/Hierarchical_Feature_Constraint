from __future__ import print_function
import argparse
import os
import random
import shutil
import time
import warnings
from tqdm import tqdm
import numpy as np
import pickle

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
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def plot_fig(key, mse_raw, array, temp_dir):
    pearsonr_value = pearsonr(mse_raw, array)[0]
    plt.scatter(mse_raw, array)
    plt.title(f'Correlation between {key} and mse_raw (of 7-th feature) : {pearsonr_value}')
    plt.savefig(os.path.join(temp_dir, key))
    plt.cla()

def get_pearsonr_mse(kd, lid, maha, src_model, temp_dir):
    key = 'try_ex1_layer7_Linf_4'
    layer_index = 7
    save_dir = os.path.join(temp_dir, key)
    data = np.load(os.path.join(temp_dir, f'{key}.npy'))[:700]
    mse = np.load(os.path.join(temp_dir, key, 'mse.npy'))[:700]
    mse_raw = np.load(os.path.join(temp_dir, key, 'mse_raw.npy'))[:700]
    plot_fig('mse', mse_raw, mse, save_dir)
    plot_fig('KD', mse_raw, kd[key]['kd_score'].transpose()[layer_index], save_dir)
    plot_fig('LID', mse_raw, lid[key][20].transpose()[layer_index], save_dir)
    plot_fig('MAHA', mse_raw, maha[key][0.0005].transpose()[layer_index], save_dir)

def run(args, default_victims):

    key_clean = 'clean'
    key_attack = args.attack
    # key_attack = 'try_ex3_layer7_Linf_4'
    # key_attack = 'try_ex1_layer7_Linf_4'
    # key_attack = 'try_ex1_layer14_Linf_4'
    # key_attack = 'try_ex1_layer2_Linf_4'
    key_noise = args.noise
    attack_methods = [key_clean, key_attack, key_noise]
    print(f"Extract features for attack: {key_attack}")

    rewrite_all = False
    temp_dir = os.path.join(os.getcwd(), f'runs_{args.dataset}', args.arch)
    print(f'temp dir {temp_dir}')
    save_dir = os.path.join(temp_dir, key_attack)
    if not os.path.isdir(save_dir): os.mkdir(save_dir)

    saver = Saver(args.arch, dataset=args.dataset)
    logging.info(f'Attacking {default_victims} by {args.arch}')
    
    train_loader = get_dataloader(dataset=args.dataset, mode='train',\
        batch_size=args.batch_size, num_workers=args.workers, \
        num_fold=args.num_fold, targeted=False)
    num_classes = train_loader.dataset.num_classes
    num_classes_selected = train_loader.dataset.num_classes_selected

    # Load models
    if 'vgg16' in args.arch:
        src_model = infer_Cls_Net_vgg(num_classes)
    elif 'resnet50' in args.arch:
        src_model = infer_Cls_Net_resnet(num_classes)
    else:
        raise NotImplementedError
    src_model = saver.load_model(src_model, args.arch)
    src_model.eval()

    # Feature num
    temp_tensor = torch.rand(2,3,299,299).cuda()
    temp_list = src_model.feature_list(temp_tensor)[1]
    num_features = len(temp_list)
    num_cnn_layer = src_model.num_cnn

    feature_list = np.empty(num_features)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
    str_index_layers = [str(i) for i in list(range(num_features))]

    # Check data
    check_data = True
    if check_data:
        data_dict = {key:np.load(os.path.join(temp_dir, key + '.npy')) for key in attack_methods}
        gt_label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        check_adv_samples(data_dict, src_model, gt_label)

    # record estimator for MAHA
    from adv_detectors import sample_estimator
    rewrite_maha_estimator = False  or rewrite_all
    save_estimator_pth = os.path.join(save_dir, 'estimator')
    precision_pth = os.path.join(save_estimator_pth, 'precision.npz')
    mean_pth = os.path.join(save_estimator_pth, 'mean.npz')
    if not os.path.isdir(save_estimator_pth): os.mkdir(save_estimator_pth)
    if not os.path.isfile(precision_pth) or rewrite_maha_estimator:
        mean, precision = sample_estimator(src_model, num_classes_selected, feature_list, train_loader)
        logging.info(f"Save mean of estimator to {mean_pth}")
        np.savez(mean_pth, **{str(i):array for i, array in enumerate(mean)})
        logging.info(f"Save precision of estimator to {precision_pth}")
        np.savez(precision_pth, **{str(i):array for i, array in enumerate(precision)})
    else:
        logging.info(f"Load mean and precision from {save_estimator_pth}")
        mean = [array for key, array in np.load(mean_pth).items()]
        precision = [array for key, array in np.load(precision_pth).items()]

    # Record LID
    save_lid_pth = os.path.join(save_dir, 'lid.pkl')
    rewrite_lid = False or rewrite_all
    if not os.path.isfile(save_lid_pth) or rewrite_lid:
        LID_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            LID_recorder[attack_name] = dict()
            LID_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import get_LID
        get_LID(src_model, LID_recorder, label, num_features)
        for key, value in LID_recorder.items():
            if key != 'clean':
                del value['features']
            del value['data']
        logging.info(f"Save LID scores to {save_lid_pth}")
        with open(save_lid_pth, 'wb') as f:
            pickle.dump(LID_recorder, f)
    else:
        logging.info(f"Load LID scores from {save_lid_pth}")
        with open(save_lid_pth, 'rb') as f:
            LID_recorder = pickle.load(f)
        
    # Record SVM
    save_svm_pth = os.path.join(save_dir, 'svm.pkl')
    rewrite_svm = False or rewrite_all
    if not os.path.isfile(save_svm_pth) or rewrite_svm:
        SVM_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            SVM_recorder[attack_name] = dict()
            SVM_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import get_svm_score
        svm_auc = get_svm_score(src_model, SVM_recorder)
        print(f'SVM score: {svm_auc}')
        for key, value in SVM_recorder.items():
            if key != 'model':
                del value['feature']
                del value['data']
        logging.info(f"Save SVM scores to {save_svm_pth}")
        with open(save_svm_pth, 'wb') as f:
            pickle.dump(SVM_recorder, f)
    else:
        logging.info(f"Load SVM scores from {save_svm_pth}")
        with open(save_svm_pth, 'rb') as f:
            SVM_recorder = pickle.load(f)

    # Record DNN
    save_dnn_pth = os.path.join(save_dir, 'dnn.pkl')
    rewrite_dnn = False or rewrite_all
    if not os.path.isfile(save_dnn_pth) or rewrite_dnn:
        DNN_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            DNN_recorder[attack_name] = dict()
            DNN_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import train_DNN_classifier
        train_DNN_classifier(src_model, DNN_recorder, num_cnn_layer=num_cnn_layer)
        logging.info(f"Save DNN scores to {save_dnn_pth}")
        with open(save_dnn_pth, 'wb') as f:
            pickle.dump(DNN_recorder, f)
    else:
        logging.info(f"Load DNN scores from {save_dnn_pth}")
        with open(save_dnn_pth, 'rb') as f:
            DNN_recorder = pickle.load(f)
    for i in range(num_cnn_layer):
        if not 'adv_pred' in DNN_recorder[i].keys():
            DNN_recorder[i]['adv_pred'] = DNN_recorder[i]['model'].infer_array(\
                torch.from_numpy(DNN_recorder[key_attack]['data']).float().cuda(), src_model)
    with open(save_dnn_pth, 'wb') as f:
        pickle.dump(DNN_recorder, f)
        
    # # Record GMM
    # save_gmm_pth = os.path.join(save_dir, 'gmm.pkl')
    # rewrite_gmm = False or rewrite_all
    # if not os.path.isfile(save_gmm_pth) or rewrite_gmm:
    #     GMM_recorder = dict()
    #     label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
    #     for attack_name in attack_methods:
    #         GMM_recorder[attack_name] = dict()
    #         GMM_recorder[attack_name]['data'] = \
    #             np.load(os.path.join(temp_dir, attack_name + '.npy'))
    #     from adv_detectors import get_GMM_scores
    #     get_GMM_scores(src_model, GMM_recorder, temp_dir, num_features, num_classes_selected)
    #     logging.info(f"Save GMM scores to {save_gmm_pth}")
    #     with open(save_gmm_pth, 'wb') as f:
    #         pickle.dump(GMM_recorder, f)
    # else:
    #     logging.info(f"Load GMM scores from {save_gmm_pth}")
    #     with open(save_gmm_pth, 'rb') as f:
    #         GMM_recorder = pickle.load(f)
    # print('\n Layer ID: {} '.format('\t'.join(str_index_layers)), end='')
    # print(f'\n GMM score :')
    # for index_layer in range(num_features):
    #     temp_aucroc = get_pairs_auc(GMM_recorder[index_layer][key_clean],
    #                     GMM_recorder[index_layer][key_noise],
    #                     GMM_recorder[index_layer][key_attack], negative=False)
    #     print('{:.3f}'.format(temp_aucroc), end='\t')
    # print('\n')

    # # Record DkNN
    # save_dknn_pth = os.path.join(save_dir, 'dknn.pkl')
    # rewrite_dknn = False or rewrite_all
    # if not os.path.isfile(save_dknn_pth) or rewrite_dknn:
    #     DkNN_recorder = dict()
    #     label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
    #     for attack_name in attack_methods:
    #         DkNN_recorder[attack_name] = dict()
    #         DkNN_recorder[attack_name]['data'] = \
    #             np.load(os.path.join(temp_dir, attack_name + '.npy'))
    #     from adv_detectors import build_dknn
    #     build_dknn(src_model, DkNN_recorder, train_loader, label, num_features=num_features)
    #     logging.info(f"Save DkNN scores to {save_dknn_pth}")
    #     with open(save_dknn_pth, 'wb') as f:
    #         pickle.dump(DkNN_recorder, f)
    # else:
    #     logging.info(f"Load DkNN scores from {save_dknn_pth}")
    #     with open(save_dknn_pth, 'rb') as f:
    #         DkNN_recorder = pickle.load(f) 
    # print('\n Layer ID: {} Logits_Regression'.format('\t'.join(str_index_layers)), end='')
    # print(f'DkNN score :\n ')
    # for index_layer in range(num_features):
    #     temp_aucroc = get_pairs_auc(DkNN_recorder[index_layer][key_clean],
    #                     DkNN_recorder[index_layer][key_noise],
    #                     DkNN_recorder[index_layer][key_attack], negative=False)
    #     print('{:.3f}'.format(temp_aucroc), end='\t')
    # final_aucroc = get_pairs_auc(DkNN_recorder[key_clean]['score'],
    #                 DkNN_recorder[key_noise]['score'],
    #                 DkNN_recorder[key_attack]['score'], negative=False)
    # print(f'DkNN score: {final_aucroc}')

    # Record MAHA
    save_maha_pth = os.path.join(save_dir, 'maha.pkl')
    rewrite_maha = False or rewrite_all
    if not os.path.isfile(save_maha_pth) or rewrite_maha:
        maha_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            maha_recorder[attack_name] = dict()
            maha_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import get_Mahalanobis_score_adv
        get_Mahalanobis_score_adv(src_model, maha_recorder, label, num_features, mean, precision)
        for key, value in maha_recorder.items():
            del value['data']
        logging.info(f"Save MAHA scores to {save_maha_pth}")
        with open(save_maha_pth, 'wb') as f:
            pickle.dump(maha_recorder, f)
    else:
        logging.info(f"Load MAHA scores from {save_maha_pth}")
        with open(save_maha_pth, 'rb') as f:
            maha_recorder = pickle.load(f)
    
    # Recored KD estimator
    save_kd_estimator_pth = os.path.join(save_dir, 'kd_estimator.pkl')
    rewrite_kd_estimator = False or rewrite_all
    if not os.path.isfile(save_kd_estimator_pth) or rewrite_kd_estimator:
        kd_estimator_record = dict()
        for attack_name in attack_methods:
            kd_estimator_record[attack_name] = dict()
        from adv_detectors import kd_estimator
        kd_estimator(src_model, kd_estimator_record, train_loader, num_classes_selected, num_features)
        logging.info(f"Save KD estimator to {save_kd_estimator_pth}")
        with open(save_kd_estimator_pth, 'wb') as f:
            pickle.dump(kd_estimator_record, f)
    else:
        logging.info(f"Load KD estimator from {save_kd_estimator_pth}")
        with open(save_kd_estimator_pth, 'rb') as f:
            kd_estimator_record = pickle.load(f)

    # Record KD score
    save_kd_pth = os.path.join(save_dir, 'kd.pkl')
    rewrite_kd = False or rewrite_all
    if not os.path.isfile(save_kd_pth) or rewrite_kd:
        kd_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            kd_recorder[attack_name] = dict()
            kd_recorder[attack_name]['kd_score'] = list()
            kd_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import get_kd_score
        get_kd_score(src_model, kd_recorder, label, num_classes_selected, kd_estimator_record, num_features)
        for key, value in kd_recorder.items():
            del value['data']
        logging.info(f"Save KD scores to {save_kd_pth}")
        with open(save_kd_pth, 'wb') as f:
            pickle.dump(kd_recorder, f)
    else:
        logging.info(f"Load KD scores from {save_kd_pth}")
        with open(save_kd_pth, 'rb') as f:
            kd_recorder = pickle.load(f)
    
    # Record BU score
    save_bu_pth = os.path.join(save_dir, 'bu.pkl')
    rewrite_bu = True or rewrite_all
    if not os.path.isfile(save_bu_pth) or rewrite_bu:
        bu_recorder = dict()
        label = np.load(os.path.join(temp_dir, 'gt' + '.npy'))
        for attack_name in attack_methods:
            bu_recorder[attack_name] = dict()
            bu_recorder[attack_name]['data'] = \
                np.load(os.path.join(temp_dir, attack_name + '.npy'))
        from adv_detectors import get_bu_scores
        get_bu_scores(src_model, bu_recorder)
        for key, value in bu_recorder.items():
            del value['data']
        logging.info(f"Save bu scores to {save_bu_pth}")
        with open(save_bu_pth, 'wb') as f:
            pickle.dump(bu_recorder, f)
    else:
        logging.info(f"Load bu scores from {save_bu_pth}")
        with open(save_bu_pth, 'rb') as f:
            bu_recorder = pickle.load(f)
    final_aucroc = get_pairs_auc(bu_recorder[key_clean]['scores'],
                    bu_recorder[key_noise]['scores'],
                    bu_recorder[key_attack]['scores'], negative=True)
    print(f'BU score: {final_aucroc}')
    # clean_score = np.stack((bu_recorder[key_clean]['scores'], kd_recorder[key_clean]['kd_score'].transpose()[num_features-1]), axis=-1)
    # noise_score = np.stack((bu_recorder[key_noise]['scores'], kd_recorder[key_noise]['kd_score'].transpose()[num_features-1]), axis=-1)
    # attack_score = np.stack((bu_recorder[key_attack]['scores'], kd_recorder[key_attack]['kd_score'].transpose()[num_features-1]), axis=-1)
    # logits_aucroc, model_lr = logits_regression_auc(clean_score, noise_score, attack_score)
    # print(f'KD_BU score: {logits_aucroc}')
    # bu_recorder['kd_bu_lr'] = model_lr
    # bu_recorder['kd_bu_clean'] = clean_score
    # bu_recorder['kd_bu_noise'] = noise_score
    # bu_recorder['kd_bu_attack'] = attack_score
    with open(save_bu_pth, 'wb') as f:
        pickle.dump(bu_recorder, f)

    # Compute ROC amd LogitsRegression
    # KD AUCROC
    print('\n Layer ID: {} Logits_Regression'.format('\t'.join(str_index_layers)), end='')
    print(f'\n KD score:')
    for index_layer in range(num_features):
        temp_aucroc = get_pairs_auc(kd_recorder[key_clean]['kd_score'].transpose()[index_layer],
                        kd_recorder[key_noise]['kd_score'].transpose()[index_layer],
                        kd_recorder[key_attack]['kd_score'].transpose()[index_layer])
        print('{:.3f}'.format(temp_aucroc), end='\t')
    logits_aucroc, model_lr = logits_regression_auc(kd_recorder[key_clean]['kd_score'],
                    kd_recorder[key_noise]['kd_score'],
                    kd_recorder[key_attack]['kd_score'])
    print('{:.3f}'.format(logits_aucroc), end='\t')
        
    # kd_aucroc = get_pairs_auc(kd_recorder[key_clean]['kd_score'],
    #                        kd_recorder[key_noise]['kd_score'],
    #                        kd_recorder[key_attack]['kd_score'])
    # print("KD AUC Score {}".format(kd_aucroc))

    # LID AUCROC
    overlap_list = LID_recorder['clean']['overlaps']
    print('\n Layer ID: {} Logits_Regression'.format('\t'.join(str_index_layers)), end='')
    for overlap in overlap_list:
        print(f'\n LID score Overlap: {overlap}')
        for index_layer in range(num_features):
            temp_aucroc = get_pairs_auc(LID_recorder[key_clean][overlap].transpose()[index_layer],
                           LID_recorder[key_noise][overlap].transpose()[index_layer],
                           LID_recorder[key_attack][overlap].transpose()[index_layer])
            print('{:.3f}'.format(temp_aucroc), end='\t')
        logits_aucroc, lid_model_lr = logits_regression_auc(LID_recorder[key_clean][overlap],
                        LID_recorder[key_noise][overlap],
                        LID_recorder[key_attack][overlap])
        print('{:.3f}'.format(logits_aucroc), end='\t')
        save_lid_lr_pth = os.path.join(save_dir, 'lid_lr.pkl')
        with open(save_lid_lr_pth, 'wb') as f:
            pickle.dump(lid_model_lr, f)
    
    # MAHA AUCROC
    m_list = maha_recorder['clean']['magnitude']
    print('\n Layer ID: {} Logits_Regression'.format('\t'.join(str_index_layers)), end='')
    for magnitude in m_list:
        print(f'\n MAHA score magnitude: {magnitude}')
        for index_layer in range(num_features):
            temp_aucroc = get_pairs_auc(maha_recorder[key_clean][magnitude].transpose()[index_layer],
                           maha_recorder[key_noise][magnitude].transpose()[index_layer],
                           maha_recorder[key_attack][magnitude].transpose()[index_layer])
            print('{:.3f}'.format(temp_aucroc), end='\t')
        logits_aucroc, maha_model_lr = logits_regression_auc(maha_recorder[key_clean][magnitude],
                        maha_recorder[key_noise][magnitude],
                        maha_recorder[key_attack][magnitude])
        print('{:.3f}'.format(logits_aucroc), end='\t')
        save_maha_lr_pth = os.path.join(save_dir, 'maha_lr.pkl')
        with open(save_maha_lr_pth, 'wb') as f:
            pickle.dump(maha_model_lr, f)
    
    # get_pearsonr_mse(kd_recorder, LID_recorder, maha_recorder, src_model, temp_dir)
    # import ipdb; ipdb.set_trace()

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
    parser.add_argument('--debug', default=None, action='store_true',
                        help='Use GPU or not')
    parser.add_argument('-f', '--num_fold', default=0, type=int,
                        help='Fold Number')
    parser.add_argument('--dataset', default='APTOS', type=str,
                        help='Fold Number')
    parser.add_argument('--attack', default='I_FGSM_Linf_4', type=str,
                        help='Fold Number') 
    parser.add_argument('--noise', default='Noise_Linf_2', type=str,
                        help='Fold Number')  
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    args = parser.parse_args()

    default_victims = {
                       'vgg16':'vgg16', 
                    #    'googlenet':'googlenet',
                    #    'inception_v3':'inception_v3',
                    #    'resnet50':'resnet50',
                       }

    run(args, default_victims)    
