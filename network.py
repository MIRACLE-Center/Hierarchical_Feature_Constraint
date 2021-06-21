import os
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
import my_models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from vgg import vgg16
from resnet import resnet50 
from utils import my_sequential


class Cls_Net(torch.nn.Module):
    def __init__(self, num_classes, arch, load_pretrain=True, keep_last=False):
        super(Cls_Net, self).__init__()
        model = models.__dict__[arch](pretrained=load_pretrain)

        if not keep_last:
            # Linear layer
            if 'resnet' in arch or 'inception' in arch or 'googlenet' in arch :
                num_feature = model.fc.weight.shape[1]
                model.fc = nn.Linear(num_feature, num_classes)
            else:
                num_feature = model.classifier[-1].weight.shape[1]
                model.classifier[-1] = nn.Linear(num_feature, num_classes)
        self.model = model
    
    def forward(self, x, get_features=False):
        if get_features:
            return self.model(x, True)
        else:
            return self.model(x)

class infer_Cls_Net_resnet(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(infer_Cls_Net_resnet, self).__init__()
        model = resnet50(num_classes=num_classes)
        self.model = model
        
        self.num_feature = 18
        self.num_cnn = 17

    def forward(self, x, get_features=False):
        if get_features:
            return self.model(x, True)
        else:
            return self.model(x)
    
    def feature_list(self, x):
        return self.model.feature_forward(x)
    
    def get_feature(self, x, feature_index):
        return self.model.feature_forward(x)[1][feature_index]
    
    def set_dropout(self, is_train=True):
        if is_train:
            self.model.dropout.train()
        else:
            self.model.dropout.eval()

class infer_Cls_Net_vgg(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(infer_Cls_Net_vgg, self).__init__()
        model = vgg16(num_classes=num_classes)
        self.model = model
        
        self.num_feature = model.num_feature

        self.num_feature = 15
        self.num_cnn = 13

    def forward(self, x, get_features=False):
        if get_features:
            return self.model(x, True)
        else:
            return self.model(x)
    
    def feature_list(self, x):
        return self.model.feature_forward(x)
    
    def get_feature(self, x, feature_index):
        return self.model.get_feature(x, feature_index)
    
    def get_feature_attack(self, input):
        assert(len(input) == 2)
        x, feature_index = input
        return self.model.get_feature(x, feature_index)

    def set_dropout(self, is_train=True):
        if is_train:
            self.train()
        else:
            self.eval()

from resnet import ResNet, Bottleneck
from torchvision.models.utils import load_state_dict_from_url
from torchvision.models.resnet import model_urls

if __name__ == "__main__":
    network = Cls_Net(2, 'resnet50').cuda()
    import ipdb; ipdb.set_trace()
    # from saver import Saver
    # saver = Saver('resnet50')
    # network = saver.load_model(network, 'resnet50')
    # test_tensor = torch.rand([2, 3, 299, 299]).cuda()
    # out, features = network.feature_list(test_tensor)
    # import ipdb; ipdb.set_trace()
    # network = Cls_Net(2, 'vgg16').cuda()
    # test = cls_vgg(2, 7).cuda()
    # network = generator_vgg(2, 7).cuda()
    test_tensor = torch.rand([2, 3, 256, 256]).cuda()
    out = network(test_tensor).cuda()
    # out_logits = test(out)
    # D = Dis_Small(512).cuda()
    # # import ipdb; ipdb.set_trace()
    # cls_out = D(out.view(2, 512, -1).mean(-1))
    # import ipdb; ipdb.set_trace()
    # out = network(test_tensor, get_features=True)
