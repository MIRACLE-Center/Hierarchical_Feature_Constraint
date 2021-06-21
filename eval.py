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
from scipy.spatial.distance import pdist, cdist, squareform

import matplotlib.pyplot as plt
from tqdm import tqdm

from adv_detectors import mle_batch, mle_batch_test

class Feature_Detector(object):
    def __init__(self, temp_dir, num_classes, num_layers, num_cnn_layers):
        assert(os.path.isdir(temp_dir))
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_cnn_layers = num_cnn_layers
        print(f"******** Load recorders from {temp_dir}")

        # Load MAHA estimator and clean scores
        save_estimator_pth = os.path.join(temp_dir, 'estimator')
        precision_pth = os.path.join(save_estimator_pth, 'precision.npz')
        mean_pth = os.path.join(save_estimator_pth, 'mean.npz')
        self.mean = [array for key, array in np.load(mean_pth).items()]
        self.precision = [array for key, array in np.load(precision_pth).items()]
        save_maha_pth = os.path.join(temp_dir, 'maha.pkl') 
        with open(save_maha_pth, 'rb') as f:
            self.maha_recorder = pickle.load(f)
        save_maha_lr_pth = os.path.join(temp_dir, 'maha_lr.pkl') 
        with open(save_maha_lr_pth, 'rb') as f:
            self.maha_lr = pickle.load(f)
        
        # Load LID recorder and clean scores
        save_lid_pth = os.path.join(temp_dir, 'lid.pkl')
        with open(save_lid_pth, 'rb') as f:
            self.LID_recorder = pickle.load(f)
        save_lid_lr_pth = os.path.join(temp_dir, 'lid_lr.pkl') 
        with open(save_lid_lr_pth, 'rb') as f:
            self.lid_lr = pickle.load(f)

        # Load SVM recorder
        save_svm_pth = os.path.join(temp_dir, 'svm.pkl')
        with open(save_svm_pth, 'rb') as f:
            self.SVM_recorder = pickle.load(f)
        
        # Load DNN recorder
        save_dnn_pth = os.path.join(temp_dir, 'dnn.pkl')
        with open(save_dnn_pth, 'rb') as f:
            self.DNN_recorder = pickle.load(f)

        # # Load DkNN recorder
        # save_dknn_pth = os.path.join(temp_dir, 'dknn.pkl')
        # with open(save_dknn_pth, 'rb') as f:
        #     self.DkNN_recorder = pickle.load(f)

        # Load BU recorder
        save_bu_pth = os.path.join(temp_dir, 'bu.pkl')
        with open(save_bu_pth, 'rb') as f:
            self.BU_recorder = pickle.load(f)
        
        # # Load GMM recorder
        # save_gmm_pth = os.path.join(temp_dir, 'gmm.pkl')
        # with open(save_gmm_pth, 'rb') as f:
        #     self.GMM_recorder = pickle.load(f)
        
        # Load KD estimator and clean scores
        save_kd_estimator_pth = os.path.join(temp_dir, 'kd_estimator.pkl')
        with open(save_kd_estimator_pth, 'rb') as f:
            kd_estimator = pickle.load(f)
        save_kd_pth = os.path.join(temp_dir, 'kd.pkl')
        with open(save_kd_pth, 'rb') as f:
            self.kd_recorder = pickle.load(f)
        self.clean_kde = kd_estimator['clean']

    
    def eval_patch(self, data, src_model, label=None):
        keys = [*self.DNN_recorder.keys()]
        key_clean, key_noise, key_adv = keys[0], keys[2], keys[1]
        batch_size = 100
        num_batches = np.ceil(data.shape[0] / batch_size).astype(int)
        full_lid = list()
        full_maha = list()
        full_dnn = list()
        full_dknn = list()
        full_bu = list()
        pred_list = list()
        data_len = data.shape[0]
        for index in range(num_batches):
            input_patch = data[index*batch_size : min(data_len, (index+1)*batch_size)]
            with torch.no_grad():
                pred = src_model(input_patch).argmax(1).cpu().numpy()
            pred_list.append(pred)
        pred_list = np.concatenate(pred_list)
        if label is not None:
            # if label.shape != 
            bingo = (pred_list == label).astype(np.float).mean()
            print(f" Attack Acc {bingo}")

        for layer_index in range(0, self.num_layers):
            src_model.eval()
            kd_result = []
            maha_result = []
            LID_result = []
            SVM_result = []
            dknn_result = []
            gmm_result = []
            maha_mean = np.array(self.mean[layer_index])
            maha_precision = torch.from_numpy(self.precision[layer_index])

            # DNN evaluate
            if layer_index < self.num_cnn_layers:
                dnn_pred = self.DNN_recorder[layer_index]['model'].infer_array(data, src_model)
                dnn_auc = get_pairs_auc(self.DNN_recorder[layer_index]['clean_pred'],
                        self.DNN_recorder[layer_index]['noise_pred'], dnn_pred, adv_test=True, negative=False)
                raw_dnn_auc = self.DNN_recorder[layer_index]['auc']
            else:
                dnn_auc = -1
                raw_dnn_auc = -1
            full_dnn.append(dnn_pred)

            for index in range(num_batches):
                data_len = data.shape[0]
                input_patch = data[index*batch_size : min(data_len, (index+1)*batch_size)]
                with torch.no_grad():
                    test_feature = src_model.get_feature(input_patch, layer_index)
                    pred = src_model(input_patch).argmax(1).cpu().numpy()

                n, c = test_feature.shape[0], test_feature.shape[1]
                test_feature = test_feature.view(n, c, -1).mean(-1).cpu().numpy()
                # pred = labels[index * batch_size: (index + 1) * batch_size]
                # KD evaluate
                for i in range(n):
                    kd_score = self.clean_kde[pred[i]][layer_index].\
                        score_samples(test_feature[i].reshape(1, -1))[0]
                    kd_result.append(kd_score)
                    # if i == 0 and layer_index>=48 and index == 1:
                    #     print(kd_score)
                    #     print(test_feature[i])
                    #     import ipdb; ipdb.set_trace()

                # LID evaluate
                clean_batch = self.LID_recorder['clean']['features'][layer_index][index]
                lid_score = mle_batch(clean_batch, test_feature, k = 20)
                # lid_score_clean = mle_batch_test(clean_batch, test_feature, clean_batch, k = 20)
                LID_result.append(lid_score)

                # MAHA evaluate
                for i in range(self.num_classes):
                    batch_sample_mean = maha_mean[i]
                    zero_f = torch.from_numpy(test_feature - batch_sample_mean)
                    term_gau = -torch.mm(torch.mm(zero_f, maha_precision), zero_f.t()).diag()
                    if i == 0:
                        noise_gaussian_score = term_gau.view(-1,1)
                    else:
                        noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      
                noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
                maha_result.append(noise_gaussian_score.numpy())
                
                # SVM evaluate
                if layer_index == self.num_layers - 1:
                    svm_scores = self.SVM_recorder['model'].predict_proba(test_feature).transpose()[1]
                    SVM_result.append(svm_scores)

                # BU evaluate
                if layer_index == self.num_layers - 1:
                    src_model.set_dropout(True)
                    bu_temp = list()
                    with torch.no_grad():
                        for _ in range(50):
                            output = src_model(input_patch)[:, 0:1].detach().cpu().numpy()
                            bu_temp.append(output)
                    bu_temp = np.concatenate(bu_temp, -1).std(-1)
                    full_bu.append(bu_temp)
                    src_model.set_dropout(False)
                
                # # GMM evaluate
                # scores = self.GMM_recorder[layer_index]['model'].score_samples(test_feature)
                # gmm_result.append(scores)

                # # DkNN evaluate
                # temp = self.DkNN_recorder[layer_index]['model'].kneighbors(test_feature)[1]
                # temp_classes = self.DkNN_recorder['labels'][temp]
                # temp_correct = (temp_classes.transpose() == pred).transpose()
                # dknn_scores = temp_correct.sum(-1) / 5
                # dknn_result.append(dknn_scores)

            # Compute AUCs
            kd_result = np.array(kd_result)
            maha_result = np.concatenate(maha_result, axis=0)
            LID_result = np.concatenate(LID_result, axis=0)
            # dknn_result = np.concatenate(dknn_result, axis=0)
            # gmm_result = np.concatenate(gmm_result, axis=0)
            # full_dknn.append(np.expand_dims(dknn_result, -1))

            kd_auc = get_pairs_auc(self.kd_recorder[key_clean]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_noise]['kd_score'].transpose()[layer_index], kd_result, adv_test=True)
            LID_auc = get_pairs_auc(self.LID_recorder[key_clean][20].transpose()[layer_index],
                        self.LID_recorder[key_noise][20].transpose()[layer_index], LID_result, adv_test=True)
            MAHA_auc = get_pairs_auc(self.maha_recorder[key_clean][0.0005].transpose()[layer_index],
                        self.maha_recorder[key_noise][0.0005].transpose()[layer_index], maha_result, adv_test=True)
            raw_kd_auc = get_pairs_auc(self.kd_recorder[key_clean]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_noise]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_adv]['kd_score'].transpose()[layer_index])
            raw_LID_auc = get_pairs_auc(self.LID_recorder[key_clean][20].transpose()[layer_index],
                        self.LID_recorder[key_noise][20].transpose()[layer_index],
                        self.LID_recorder[key_adv][20].transpose()[layer_index])
            raw_MAHA_auc = get_pairs_auc(self.maha_recorder[key_clean][0.0005].transpose()[layer_index],
                        self.maha_recorder[key_noise][0.0005].transpose()[layer_index],
                        self.maha_recorder[key_adv][0.0005].transpose()[layer_index])
            # dknn_auc = get_pairs_auc(self.DkNN_recorder[layer_index][key_clean],
                        # self.DkNN_recorder[layer_index][key_noise], dknn_result, adv_test=True, negative=False)
            # GMM_auc = get_pairs_auc(self.GMM_recorder[layer_index][key_clean],
            #             self.GMM_recorder[layer_index][key_noise], gmm_result, adv_test=True, negative=False)
            
            print('Layer {} \t KD {:.3f}/{:.3f} \t LID {:.3f}/{:.3f} \t MAHA {:.3f}/{:.3f} \t DNN {:.3f}/{:.3f} \t'.\
                format(layer_index, raw_kd_auc, kd_auc, raw_LID_auc, LID_auc, \
                    raw_MAHA_auc, MAHA_auc, raw_dnn_auc, dnn_auc))
            
            full_lid.append(np.expand_dims(LID_result, 0))
            full_maha.append(np.expand_dims(maha_result, 0))

            if layer_index == self.num_layers - 1:
                kd_auc, kd_rate = get_pairs_auc(self.kd_recorder[key_clean]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_noise]['kd_score'].transpose()[layer_index], kd_result, adv_test=True, get_rate=True)
                raw_kd_auc, raw_kd_rate = get_pairs_auc(self.kd_recorder[key_clean]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_noise]['kd_score'].transpose()[layer_index],
                        self.kd_recorder[key_adv]['kd_score'].transpose()[layer_index], get_rate=True)
                print('Final KD AUC {:.3f}/{:.3f} TNR at 90 {:.3f}/{:.3f}'.\
                    format(raw_kd_auc, kd_auc, raw_kd_rate, kd_rate))
        
        full_lid = np.concatenate(full_lid, 0).transpose()
        full_maha = np.concatenate(full_maha, 0).transpose()
        full_svm = np.concatenate(SVM_result)
        # full_dknn = np.concatenate(full_dknn, -1).sum(-1)

        # import ipdb; ipdb.set_trace()
        final_maha_auc, maha_rate = logits_regression_infer(self.maha_recorder[key_clean][0.0005],\
            self.maha_recorder[key_noise][0.0005], full_maha, self.maha_lr, get_rate=True)
        final_raw_maha_auc, raw_maha_rate = logits_regression_infer_raw(self.maha_recorder[key_clean][0.0005],\
            self.maha_recorder[key_noise][0.0005], \
            self.maha_recorder[key_adv][0.0005], self.maha_lr, get_rate=True)
        print('Final MAHA AUC {:.3f}/{:.3f} TNR at 90 {:.3f}/{:.3f}'.\
            format(final_raw_maha_auc, final_maha_auc, raw_maha_rate, maha_rate))
        # import ipdb; ipdb.set_trace()

        final_lid_auc, lid_rate = logits_regression_infer(self.LID_recorder[key_clean][20],\
            self.LID_recorder[key_noise][20], full_lid, self.lid_lr, get_rate=True)
        final_raw_lid_auc, raw_lid_rate = logits_regression_infer_raw(self.LID_recorder[key_clean][20],\
            self.LID_recorder[key_noise][20],\
            self.LID_recorder[key_adv][20], self.lid_lr, get_rate=True)
        print('Final LID AUC {:.3f}/{:.3f} TNR at 90 {:.3f}/{:.3f}'.\
            format(final_raw_lid_auc, final_lid_auc, raw_lid_rate, lid_rate))

        final_svm_auc, svm_rate = get_pairs_auc(self.SVM_recorder[key_clean]['scores'],\
            self.SVM_recorder[key_noise]['scores'], full_svm, adv_test=True, negative=False, get_rate=True)
        final_raw_svm_auc, raw_svm_rate = get_pairs_auc(self.SVM_recorder[key_clean]['scores'],\
            self.SVM_recorder[key_noise]['scores'],
            self.SVM_recorder[key_adv]['scores'], negative=False, get_rate=True)
        print('Final SVM AUC {:.3f}/{:.3f} TNR at 90 {:.3f}/{:.3f}'.\
            format(final_raw_svm_auc, final_svm_auc, raw_svm_rate, svm_rate))

        full_dnn = np.stack(full_dnn, -1).mean(-1)
        ensamble_clean = [self.DNN_recorder[id]['clean_pred'] for id in range(self.num_cnn_layers)]
        ensamble_clean = np.stack(ensamble_clean, -1).mean(-1)
        ensamble_noise = [self.DNN_recorder[id]['noise_pred'] for id in range(self.num_cnn_layers)]
        ensamble_noise = np.stack(ensamble_noise, -1).mean(-1)
        ensamble_adv = [self.DNN_recorder[id]['adv_pred'] for id in range(self.num_cnn_layers)]
        ensamble_adv = np.stack(ensamble_adv, -1).mean(-1)
        final_dnn_auc, dnn_rate = get_pairs_auc(ensamble_clean, ensamble_noise, full_dnn, adv_test=True, negative=False, get_rate=True)
        final_raw_dnn_auc, raw_dnn_rate = get_pairs_auc(ensamble_clean, ensamble_noise, ensamble_adv, negative=False, get_rate=True)
        print('Final DNN AUC {:.3f}/{:.3f} (After ensamlbe)  TNR at 90 {:.3f}/{:.3f}'.\
            format(final_raw_dnn_auc, final_dnn_auc, raw_dnn_rate, dnn_rate))

        # final_dknn_auc = get_pairs_auc(self.DkNN_recorder[key_clean]['score'],\
        #     self.DkNN_recorder[key_noise]['score'], full_dknn, adv_test=True, negative=False)
        # print('Final DkNN AUC {:.3f}'.format(final_dknn_auc))

        full_bu = np.concatenate(full_bu)
        final_bu_auc, bu_rate = get_pairs_auc(self.BU_recorder[key_clean]['scores'],\
            self.BU_recorder[key_noise]['scores'], full_bu, adv_test=True, get_rate=True)
        final_raw_bu_auc, raw_bu_rate = get_pairs_auc(self.BU_recorder[key_clean]['scores'],\
            self.BU_recorder[key_noise]['scores'], self.BU_recorder[key_adv]['scores'], get_rate=True)
        print('Final BU AUC {:.3f}/{:.3f} TNR at 90 {:.3f}/{:.3f}'.\
            format(final_raw_bu_auc, final_bu_auc, raw_bu_rate, bu_rate))

        # # Last layer for GMM_auc
        # print('Final GMM AUC {:.3f}'.format(GMM_auc))
        # final_bu_kd_auc = logits_regression_infer(self.BU_recorder['kd_bu_clean'],\
        #     self.BU_recorder['kd_bu_noise'], np.stack((full_bu, kd_result), -1), \
        #     self.BU_recorder['kd_bu_lr'])
        # print('Final KD_BU AUC {:.3f}'.format(final_bu_kd_auc))
        # import ipdb; ipdb.set_trace() 


if __name__ == "__main__":
    num_classes = 2
    arch = 'resnet50'
    # arch = 'vgg16'
    root_dir = f'/home1/qsyao/Code_HFC/runs_APTOS/{arch}'
    if arch == 'vgg16':
        src_model = infer_Cls_Net_vgg(num_classes)
    elif arch == 'resnet50':
        src_model = infer_Cls_Net_resnet(num_classes)
    else:
        raise NotImplementedError
    saver = Saver(arch)
    src_model = saver.load_model(src_model, arch)
    src_model.eval()
    src_model = src_model.cuda()
    key = 'I_FGSM_Linf_1'
    # key = 'CW_Linf_2'
    temp_dir = os.path.join(root_dir, key)
    test = Feature_Detector(temp_dir, num_classes, src_model.num_feature, src_model.num_cnn)

    # test_feature_0 = np.load('/apdcephfs/share_1290796/qingsongyao/temp/APTOS/distribution/clean_0.npy').squeeze().transpose()
    # test_feature_1 = np.load('/apdcephfs/share_1290796/qingsongyao/temp/APTOS/distribution/clean_1.npy').squeeze().transpose()
    # test_feature_0 = np.load('/apdcephfs/share_1290796/qingsongyao/temp/APTOS/distribution/adv_0.npy').squeeze().transpose().astype(np.float32)    
    # test_feature_1 = np.load('/apdcephfs/share_1290796/qingsongyao/temp/APTOS/distribution/adv_1.npy').squeeze().transpose().astype(np.float32)    
    # test_feature = np.concatenate([test_feature_0, test_feature_1], axis=0)
    # labels = np.concatenate([np.zeros(test_feature_0.shape[0]), np.ones(test_feature_1.shape[1])])
    # test.eval_patch(test_feature, labels)

    # test_data = f'/apdcephfs/share_1290796/qingsongyao/temp/APTOS/vgg16/I_FGSM_Linf_4/GMM_2.npy'
    # test_data = f'/apdcephfs/share_1290796/qingsongyao/temp/APTOS/{arch}/CW_L2_1.npy'

    # test_data = f'/apdcephfs/share_1290796/qingsongyao/temp/APTOS/{arch}/CW_Linf_2.npy'
    # test_data = f'/apdcephfs/share_1290796/qingsongyao/temp/APTOS/{arch}/CW_Linf_2.npy'
    # array = np.load(test_data)[-220:]
    test_data = f'/home1/qsyao/Code_HFC/runs_APTOS/{arch}/{key}/GMM_2.npy'
    # test_data = f'/apdcephfs/share_1290796/qingsongyao/temp/APTOS/{arch}/I_FGSM_Linf_4/GMM_2.npy'
    array = np.load(test_data)
    label_pth = f'/home1/qsyao/Code_HFC/runs_APTOS/{arch}/{key}/GMM_2_pred.npy'
    labels = np.load(label_pth)
    print(array.shape)
    test_data = torch.from_numpy(array).cuda()

    
    # num_batch = test_data.shape
    # feature_dict = {i:[] for i in range(15)}
    # num_batches = int(test_data.shape[0] // 100)
    # for i in range(num_batches):
    #     input_patch = test_data[i*100 : (i+1)*100]
    #     # temp = src_model.get_feature(input_patch, 7)
    #     _, temp = src_model.feature_list(input_patch)
    #     for j in range(15):
    #         feature = temp[j]
    #         n, c = feature.shape[0], feature.shape[1]
    #         feature = feature.view(n, c, -1).mean(-1)
    #         feature_dict[j].append(feature.detach().cpu().numpy())
    # features = [np.concatenate(feature_dict[i], 0) for i in range(15)]
    test.eval_patch(test_data, src_model, labels)

