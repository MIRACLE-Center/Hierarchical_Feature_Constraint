from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
from tqdm import tqdm


# lid of a batch of query points X
def mle_batch(data, batch, k):
    '''
    commpute lid score using data & batch with k-neighbors
    return: a: computed LID score
    '''
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1]))
    # f = lambda v: v.mean()
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    # import ipdb; ipdb.set_trace()
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

# this function is from https://github.com/xingjunm/lid_adversarial_subspace_detection
def merge_and_generate_labels(X_pos, X_neg):
    """
    merge positve and nagative artifact and generate labels
    return: X: merged samples, 2D ndarray
             y: generated labels (0/1): 2D ndarray same size as X
    """
    X_pos = np.asarray(X_pos, dtype=np.float32)
    X_pos = X_pos.reshape((X_pos.shape[0], -1))

    X_neg = np.asarray(X_neg, dtype=np.float32)
    X_neg = X_neg.reshape((X_neg.shape[0], -1))

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))
    y = y.reshape((X.shape[0], 1))

    return X, y

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance
    
    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)
    
    from tqdm import tqdm
    for data, target in tqdm(train_loader, desc='Record train features'):
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad():
            output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)
            
        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()
        
        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        for j in range(num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
        
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)
                
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    for i in range(len(sample_class_mean)):
        sample_class_mean[i] = sample_class_mean[i].cpu().numpy()
        precision[i] = precision[i].cpu().numpy()

    return sample_class_mean, precision

def get_Mahalanobis_score(model, test_loader, num_classes, outf, out_flag, net_type, sample_mean, precision, layer_index, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index
    '''
    model.eval()
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer_index))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer_index))
        
    g = open(temp_file_name, 'w')
    
    for data, target in test_loader:
        
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad = True), Variable(target)
        
        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, -magnitude, gradient)
 
        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis

def get_posterior(model, net_type, test_loader, magnitude, temperature, outf, out_flag):
    '''
    Compute the maximum value of (processed) posterior distribution - ODIN
    return: null
    '''
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total = 0
    if out_flag == True:
        temp_file_name_val = '%s/confidence_PoV_In.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_In.txt'%(outf)
    else:
        temp_file_name_val = '%s/confidence_PoV_Out.txt'%(outf)
        temp_file_name_test = '%s/confidence_PoT_Out.txt'%(outf)
        
    g = open(temp_file_name_val, 'w')
    f = open(temp_file_name_test, 'w')
    
    for data, _ in test_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, requires_grad = True)
        batch_output = model(data)
            
        # temperature scaling
        outputs = batch_output / temperature
        labels = outputs.data.max(1)[1]
        labels = Variable(labels)
        loss = criterion(outputs, labels)
        loss.backward()
         
        # Normalizing the gradient to binary in {0, 1}
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))

        tempInputs = torch.add(data.data,  -magnitude, gradient)
        outputs = model(Variable(tempInputs, volatile=True))
        outputs = outputs / temperature
        soft_out = F.softmax(outputs, dim=1)
        soft_out, _ = torch.max(soft_out.data, dim=1)
        
        for i in range(data.size(0)):
            if total <= 1000:
                g.write("{}\n".format(soft_out[i]))
            else:
                f.write("{}\n".format(soft_out[i]))
                
    f.close()
    g.close()

from attackers import image_net_std
from tqdm import tqdm
image_net_std = torch.from_numpy(np.array(image_net_std)).cuda()
def get_Mahalanobis_score_adv(model, maha_recorder, test_label, num_output, mean, precision):
    '''
    Compute the proposed Mahalanobis confidence score on adversarial samples
    # num_output : layer_index
    '''
    model.eval()
    batch_size = 10
    total = 0
    num_classes = mean[0].shape[0]
    data_len = maha_recorder['clean']['data'].shape[0]

    # Convert to tensor cuda
    for i in range(len(mean)):
        mean[i] = torch.from_numpy(mean[i]).cuda()
    for i in range(len(precision)):
        precision[i] = torch.from_numpy(precision[i]).cuda()

    # m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    m_list = [0.0005]
    m_list.sort()
    for key, value in maha_recorder.items():
        for i in m_list:
            maha_recorder[key][i] = {layer_index:[] for layer_index in range(num_output)}
        maha_recorder[key]['magnitude'] = m_list

    for key, value in maha_recorder.items():

        value['guassian_scores'] = dict()
        # for i in range(num_output):
        #     value['guassian_scores'][i] = list()
        
        # Store features
        total = 0
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get MAHA for {key}"):
            data = torch.from_numpy(value['data'][total : min(total + batch_size, data_len)]).cuda().float()
            data.requires_grad = True
            target = torch.from_numpy(test_label[total : min(total + batch_size, data_len)]).cuda()
            total += batch_size
            
            for i in range(num_output):
                with torch.no_grad():
                    out_features = model.get_feature(data, i)

                # i: Layer index
                out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
                out_features = torch.mean(out_features.data, 2)

                # Get pred
                gaussian_score = 0
                for id_class in range(num_classes):
                    batch_sample_mean = mean[i][id_class]
                    zero_f = out_features.data - batch_sample_mean
                    term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[i]), zero_f.t()).diag()
                    if id_class == 0:
                        gaussian_score = term_gau.view(-1,1)
                    else:
                        gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
                sample_pred = gaussian_score.max(1)[1]
                bingo = (target == sample_pred).sum()
                
                batch_sample_mean = mean[i].index_select(0, sample_pred)
                # get target feature
                data.grad = None
                del out_features
                target_features = model.get_feature(data, i)
                torch.cuda.empty_cache()

                target_features = target_features.view(target_features.shape[0],\
                     target_features.shape[1], -1).mean(-1)
                zero_f = target_features - batch_sample_mean
                pure_gau = torch.mm(torch.mm(zero_f, precision[i]), zero_f.t()).diag()
                loss = torch.mean(pure_gau)
                loss.backward()

                # add sign grad
                for magnitude in m_list:
                    new_data = torch.zeros_like(data)
                    gradient = data.grad.sign()
                    new_data = data.data + magnitude * gradient

                    with torch.no_grad():
                        target_features = model.get_feature(new_data.float(), i)
                        target_features = target_features.view(target_features.shape[0],\
                             target_features.shape[1], -1).mean(-1)
                    noise_gaussian_score = 0
                    for id_class in range(num_classes):
                        batch_sample_mean = mean[i][id_class]
                        zero_f = target_features.data - batch_sample_mean
                        term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[i]), zero_f.t()).diag()
                        if id_class == 0:
                            noise_gaussian_score = term_gau.view(-1,1)
                        else:
                            noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)
                    noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
                    value[magnitude][i].append(noise_gaussian_score.cpu().numpy())

        for key in m_list:
            item = value[key]
            # multi layer, each layer: arrays
            layers_maha = [np.expand_dims(np.concatenate(item[i], axis=-1), -1) for i in range(num_output)]
            value[key] = np.concatenate(layers_maha, axis=-1)


def get_LID(model, LID_recorder, test_label, num_output):
    '''
    Compute LID score on adversarial samples
    return: LID score
    '''
    model.eval()  
    batch_size = 10
    
    # overlap_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    overlap_list = [20]
    for key, value in LID_recorder.items():
        for i in overlap_list:
            LID_recorder[key][i] = list()
        LID_recorder[key]['overlaps'] = overlap_list
    
    for key, value in LID_recorder.items():

        value['features'] = dict()
        for i in range(num_output):
            value['features'][i] = list()
        
        # Store features
        total = 0
        data_len = value['data'].shape[0]
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get LID for {key}"):
            data = torch.from_numpy(value['data'][total : min(data_len, total + batch_size)]).cuda().float()
            target = torch.from_numpy(test_label[total : min(data_len, total + batch_size)]).cuda()
            total += batch_size

            with torch.no_grad():
                output, out_features = model.feature_list(data)
            
            for i in range(num_output):
                out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                out_features[i] = torch.mean(out_features[i].data, 2)
                out_features[i] = out_features[i].cpu().numpy()
                value['features'][i].append(np.asarray(out_features[i], dtype=np.float32).reshape((out_features[i].shape[0], -1)))

        # Get LID scores for each layer
        for overlap in overlap_list:
            temp = list()
            for i in range(num_output):
                temp_layer = list()
                for index, item in enumerate(value['features'][i]):
                    clean_batch = LID_recorder['clean']['features'][i][0]
                    target_batch = item
                    lid_score = mle_batch(clean_batch, target_batch, k = overlap)
                    lid_score = lid_score.reshape((lid_score.shape[0], -1))
                    temp_layer.append(lid_score)
                temp_layer = np.concatenate(temp_layer, axis=0)
                temp.append(temp_layer)
            value[overlap] = np.concatenate(temp, axis=1)

from sklearn.neighbors import KernelDensity
def kd_estimator(model, kd_estimator_record, train_loader, num_classes, num_output):
    # Expand original KD to multi-layer KD
    # Also, follow code of MAHA use mean value for each featuremap
    
    model.eval()  
    value = kd_estimator_record['clean']
    for id_class in range(num_classes):
        value[id_class] = {i:[] for i in range(num_output)}
    
    # Store features
    total = 0
    for data, target in tqdm(train_loader, desc=f"Get KD estimetor"):
        total += data.size(0)
        data = data.cuda()
        with torch.no_grad():
            output, out_features = model.feature_list(data)
        for i in range(data.shape[0]):
            for layer_index in range(num_output):
                c = out_features[layer_index][i].shape[0]
                tmp_feature = out_features[layer_index][i].view(c, -1).mean(-1)
                value[target[i].item()][layer_index].append(\
                    tmp_feature.unsqueeze(0).cpu().numpy())

    for i in range(num_classes):
        for layer_index in range(num_output):
            features = np.concatenate(value[i][layer_index], 0)
            kde = KernelDensity().fit(features)
        
            value[i][layer_index] = kde

    value['num_classes'] = num_classes
        
def get_kd_score(model, KD_recorder, test_label, num_classes, kd_estimator_record, num_output):
    model.eval()
    batch_size = 10
    clean_kdes = kd_estimator_record['clean']

    for key, value in KD_recorder.items():
        for i in range(num_output):
            value[i] = list()

    for key, value in KD_recorder.items():
        total = 0
        data_len = value['data'].shape[0]
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get KD for {key}"):
            data = torch.from_numpy(value['data'][total : min(data_len, total + batch_size)]).cuda().float()
            data.requires_grad = True
            target = torch.from_numpy(test_label[total : min(data_len, total + batch_size)]).cuda()
            total += batch_size
            
            with torch.no_grad():
                output, out_features = model.feature_list(data)
                for layer_index in range(num_output):
                    output_id = output.argmax(dim=1).detach().cpu().numpy()
                    n, c = out_features[layer_index].shape[0], out_features[layer_index].shape[1]
                    tmp_feature = out_features[layer_index].view(n, c, -1).mean(-1).cpu().numpy()
            
                    for id in range(n):
                        kd_score = clean_kdes[output_id[id]][layer_index]\
                            .score_samples(tmp_feature[id].reshape(1, -1))[0]
                        value[layer_index].append(kd_score)
        
        result_scores = list()
        for layer_index in range(num_output):
            value[layer_index] = np.array(value[layer_index])
            result_scores.append(np.expand_dims(value[layer_index], -1))
        value['kd_score'] = np.concatenate(result_scores, -1)
        # print(value['kd_score'].mean())
                
def get_svm_score(model, svm_recorder):
    model.eval()
    batch_size = 10
    split = 0.3

    for key, value in svm_recorder.items():
        value['scores'] = list()
        value['feature'] = list()

    for key, value in svm_recorder.items():
        total = 0
        data_len = value['data'].shape[0]
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get SVM for {key}"):
            data = torch.from_numpy(value['data'][total : min(data_len, total + batch_size)]).cuda().float()
            data.requires_grad = True
            total += batch_size
            
            with torch.no_grad():
                output, out_features = model.feature_list(data)
                last_feature = out_features[-1].detach().cpu().numpy()
                value['feature'].append(last_feature)
        
        value['feature'] = np.concatenate(value['feature'])
    
    keys = [*svm_recorder.keys()]
    key_clean, key_noise, key_adv = keys[0], keys[2], keys[1]
    num_test = int(value['feature'].shape[0] * split)
    total_num = value['feature'].shape[0]
    num_train = total_num - num_test
    
    X_train = np.concatenate([svm_recorder[key_clean]['feature'][:-num_test], \
        svm_recorder[key_noise]['feature'][:-num_test],\
        svm_recorder[key_adv]['feature'][:-num_test]])
    Y_train = np.concatenate([np.ones(num_train), np.ones(num_train), np.zeros(num_train)])

    X_test = np.concatenate([svm_recorder[key_clean]['feature'][-num_test:], \
        svm_recorder[key_noise]['feature'][-num_test:],\
        svm_recorder[key_adv]['feature'][-num_test:]])
    Y_test = np.concatenate([np.ones(num_test), np.ones(num_test), np.zeros(num_test)])

    # Train RBF-SVM
    from sklearn.svm import SVC
    rbf_svc = SVC(kernel='rbf', probability=True)
    rbf_svc.fit(X_train, Y_train)

    prediction = rbf_svc.predict_proba(X_test).transpose()[1]
    from sklearn.metrics import roc_auc_score
    svm_auc = roc_auc_score(Y_test, prediction)
    for key, value in svm_recorder.items():
        value['scores'] = rbf_svc.predict_proba(value['feature']).transpose()[1]

    svm_recorder['model'] = rbf_svc

    return svm_auc

from sklearn.metrics import roc_auc_score
class DNN_CNN_Feature(torch.nn.Module):
    
    def __init__(self, num_channel, train_data, train_label, \
                infer_model, id_layer, test_set, clean_data, noise_data, adv_data):
        super(DNN_CNN_Feature, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channel, 96, kernel_size=3, padding=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 192, kernel_size=3, padding=4),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 192, kernel_size=3, padding=4),
            nn.Conv2d(192, 1, kernel_size=3, padding=4),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Sigmoid()
        )

        self.model.cuda()
        self.model.train()
        self.id_layer = id_layer
        infer_model = infer_model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 1e-3,
                                weight_decay=1e-4)

        # for i in range(500):
        for i in tqdm(list(range(200)), desc=f'Training DNN for layer {id_layer}'):
            ids = torch.randint(train_data.shape[0], (64, 1)).squeeze()
            patch_data, patch_label = train_data[ids], train_label[ids]
            with torch.no_grad():
                input_feature = infer_model.get_feature(patch_data, id_layer)
            pred = self.model(input_feature).squeeze()
            loss_bce = loss_fn(pred, patch_label)
            optimizer.zero_grad()
            loss_bce.backward()
            optimizer.step()
            # if (i + 1) % 100 == 0:
            #     print(self.infer_auc(test_set, infer_model))
            # import ipdb; ipdb.set_trace()
        
        self.final_auc = self.infer_auc(test_set, infer_model)
        self.clean_pred = self.infer_array(clean_data, infer_model)
        self.noise_pred = self.infer_array(noise_data, infer_model)
        self.adv_pred = self.infer_array(adv_data, infer_model)
        self.model = self.model.cpu()
        print('DNN Cls Layer {} AUC {:.3f}'.format(id_layer, self.final_auc))
        
    def infer_auc(self, test_set, infer_model):
        batch_size = 10
        num_data = test_set[0].shape[0]
        num_batch = num_data // batch_size
        infer_out = list()
        self.model.eval()
        for index in range(num_batch + 1):
            batch_data = test_set[0][index*batch_size : min(num_data, (index+1)*batch_size)]
            if batch_data.shape[0] == 0: break
            with torch.no_grad():
                batch_feature = infer_model.get_feature(batch_data, self.id_layer)
                pred = self.model(batch_feature).squeeze()
            infer_out.append(pred.cpu().numpy())
        infer_out = np.concatenate(infer_out)
        return roc_auc_score(test_set[1].cpu().int().numpy(), infer_out)
    
    def infer_array(self, input_array, infer_model):
        batch_size = 10
        num_data = input_array.shape[0]
        num_batch = num_data // batch_size
        infer_out = list()
        self.model.eval()
        self.model = self.model.cuda()
        for index in range(num_batch + 1):
            batch_data = input_array[index*batch_size : min(num_data, (index+1)*batch_size)]
            if batch_data.shape[0] == 0: break
            with torch.no_grad():
                batch_feature = infer_model.get_feature(batch_data, self.id_layer)
                pred = self.model(batch_feature).squeeze()
            infer_out.append(pred.cpu().numpy())
        infer_out = np.concatenate(infer_out)
        return infer_out

def train_DNN_classifier(model, DNN_recorder, num_cnn_layer=13):
    model = model.cuda()
    model.eval()

    keys = [*DNN_recorder.keys()]
    key_clean, key_noise, key_adv = keys[0], keys[2], keys[1]

    test_data = torch.from_numpy(DNN_recorder[key_clean]['data'][:2]).float().cuda()
    test_features = model.feature_list(test_data)[1][:num_cnn_layer]
    num_channels = [feature.shape[1] for feature in test_features]
    DNN_recorder['num_cnn_layer'] = num_cnn_layer

    num_train_clean = int(DNN_recorder[key_clean]['data'].shape[0] * 0.7)
    num_train_noise = int(DNN_recorder[key_noise]['data'].shape[0] * 0.7)
    num_train_adv = int(DNN_recorder[key_adv]['data'].shape[0] * 0.7)
    num_test_clean = DNN_recorder[key_clean]['data'].shape[0] - num_train_clean
    num_test_noise = DNN_recorder[key_noise]['data'].shape[0] - num_train_noise
    num_test_adv = DNN_recorder[key_adv]['data'].shape[0] - num_train_adv

    train_data = np.concatenate([DNN_recorder[key_clean]['data'][:num_train_clean],
                          DNN_recorder[key_noise]['data'][:num_train_noise],
                          DNN_recorder[key_adv]['data'][:num_train_adv]])
    train_label = np.concatenate([np.ones(num_train_clean),
                           np.ones(num_train_noise),
                           np.zeros(num_train_adv)])
    train_data = torch.from_numpy(train_data).float().cuda()
    train_label = torch.from_numpy(train_label).float().cuda()

    test_data = np.concatenate([DNN_recorder[key_clean]['data'][num_train_clean:],
                          DNN_recorder[key_noise]['data'][num_train_noise:],
                          DNN_recorder[key_adv]['data'][num_train_adv:]])
    test_label = np.concatenate([np.ones(num_test_clean),
                           np.ones(num_test_noise),
                           np.zeros(num_test_adv)])
    test_data = torch.from_numpy(test_data).float().cuda()
    test_label = torch.from_numpy(test_label).float().cuda()

    for id_layer in range(0, num_cnn_layer):
        cls_model = DNN_CNN_Feature(\
            num_channels[id_layer], train_data, \
            train_label, model, id_layer, [test_data, test_label],\
            torch.from_numpy(DNN_recorder[key_clean]['data']).float().cuda(),\
            torch.from_numpy(DNN_recorder[key_noise]['data']).float().cuda(),\
            torch.from_numpy(DNN_recorder[key_adv]['data']).float().cuda())
        DNN_recorder[id_layer] = dict()
        DNN_recorder[id_layer]['model'] = cls_model
        DNN_recorder[id_layer]['auc'] = cls_model.final_auc
        DNN_recorder[id_layer]['clean_pred'] = cls_model.clean_pred
        DNN_recorder[id_layer]['noise_pred'] = cls_model.noise_pred
        DNN_recorder[id_layer]['adv_pred'] = cls_model.adv_pred

def get_bu_scores(scr_model, bu_recorder):
    scr_model.set_dropout(True)
    batch_size = 10

    for key, value in bu_recorder.items():
        value['scores'] = list()

    for key, value in bu_recorder.items():
        total = 0
        data_len = value['data'].shape[0]
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get BU for {key}"):
            data = torch.from_numpy(value['data'][total : min(data_len, total + batch_size)]).cuda().float()
            total += batch_size
            
            with torch.no_grad():
                results = list()
                for i in range(50):
                    output = scr_model(data)
                    # output = torch.nn.functional.softmax(output, -1)
                    results.append(output[:, 0:1].detach().cpu().numpy())
                results = np.concatenate(results, -1)
            value['scores'].append(results.std(-1))
        value['scores'] = np.concatenate(value['scores'])
        # import ipdb; ipdb.set_trace()

import pickle
def get_GMM_scores(scr_model, GMM_recorder, root_dir, num_layers=15, num_classes=2):
    # Need to gemm GMM models first in gen_GMM.py

    scr_model = scr_model.cuda()
    scr_model.eval()

    keys = [*GMM_recorder.keys()]
    key_clean, key_noise, key_adv = keys[0], keys[2], keys[1]

    for _ in range(num_layers):
        GMM_recorder[_] = dict()
        for key in keys:
            GMM_recorder[_][key] = list()

    gmm_model_pth = os.path.join(root_dir, 'GMM')
    for id_layer in range(num_layers):
        for id_class in range(num_classes):
            with open(os.path.join(gmm_model_pth, f'Layer_{id_layer}_class_{id_class}.pkl'), 'rb') as f:
                gmm_model = pickle.load(f)
            GMM_recorder[id_layer]['model'] = gmm_model
    
    batch_size = 10
    for key in keys:
        value = GMM_recorder[key]
        total = 0
        data_len = value['data'].shape[0]
        for data_index in tqdm(range(int(np.ceil(value['data'].shape[0]/batch_size))), desc=f"Get GMM for {key}"):
            data = torch.from_numpy(value['data'][total : min(data_len, total + batch_size)]).cuda().float()
            total += batch_size
            
            with torch.no_grad():
                _, features = scr_model.feature_list(data)
                for id_layer in range(num_layers):
                    n, c = features[id_layer].shape[0], features[id_layer].shape[1]
                    temp = features[id_layer].view(n, c, -1).mean(-1).cpu().numpy()
                    scores = GMM_recorder[id_layer]['model'].score_samples(temp)
                    # probs = GMM_recorder[id_layer]['model'].predict_proba(temp)
                    # import ipdb; ipdb.set_trace()
                    GMM_recorder[id_layer][key].append(scores)
        for id_layer in range(num_layers):
            GMM_recorder[id_layer][key] = np.concatenate(GMM_recorder[id_layer][key])

