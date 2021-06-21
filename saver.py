import os
import logging
import torch
import pickle
import numpy as np
import torch
import torchvision

import cv2
from PIL import Image
from utils import gen_results_strings


def normalize(tensor):
    tensor = tensor.mul(attackers.torch_std_cuda)\
        .add(attackers.torch_mean_cuda)
    return tensor

def clip_01(tensor):
    return torch.clamp(tensor, 0, 1).unsqueeze(0)

def norm_01(tensor):
    out = np.clip(tensor, -1, 1)
    return (out + 1) / 2 

def norm_pertubation(tensor):
    tensor = tensor * (64 / 1)
    tensor = tensor + 0.5
    return tensor

class Saver(object):
    def __init__(self, name, dataset='APTOS'):
        self.name = name
        self.root_dir = os.path.join(os.getcwd(), f'runs_{dataset}')
        self.root_pth = os.path.join(self.root_dir, name)
        self.ckpt_pth = os.path.join(self.root_pth, 'checkpoints')
        self.visual_pth = os.path.join(self.root_pth, 'visual')
        self.attack_pth = os.path.join(self.root_pth, 'attack')
        self.attack_visual_pth = os.path.join(self.attack_pth, 'visual')
    
        if not os.path.exists(self.root_pth): os.mkdir(self.root_pth)
        if not os.path.exists(self.ckpt_pth): os.mkdir(self.ckpt_pth)
        if not os.path.exists(self.visual_pth): os.mkdir(self.visual_pth)
        if not os.path.exists(self.attack_pth): os.mkdir(self.attack_pth)
        if not os.path.exists(self.attack_visual_pth): os.mkdir(self.attack_visual_pth)
    
        log_pth = os.path.join(self.root_pth, 'log.log')
        logging.basicConfig(filename=log_pth, filemode="a", \
            format="%(asctime)s %(name)s:%(levelname)s:%(message)s", \
                datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)

    def mkdir(self, name):
        pth = os.path.join(self.root_pth, name)
        if not os.path.exists(pth): os.mkdir(pth)
        return pth

    def save_best_model(self, model):
        ckpt_pth = os.path.join(self.ckpt_pth, 'best_model.pth')
        torch.save(model.state_dict(), ckpt_pth)
    
    def save_current_model(self, model):
        ckpt_pth = os.path.join(self.ckpt_pth, 'current_model.pth')
        torch.save(model.state_dict(), ckpt_pth)
    
    def save_epoch_model(self, model, epoch):
        ckpt_pth = os.path.join(self.ckpt_pth, f'{epoch}.pth')
        torch.save(model.state_dict(), ckpt_pth)
    
    def load_epoch_model(self, model, epoch):
        ckpt_pth = os.path.join(self.ckpt_pth, f'{epoch}.pth')
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt)
        print(f"Load {ckpt_pth}")
        return model
    
    def save_metrics(self, **kwargs):
        result_pth = os.path.join(self.root_pth, 'results.txt')
        with open(result_pth, 'w') as f:
            f.write(f'Results for {self.name}: \n')
            for key, value in kwargs.items():
                f.write(f'{key}:{value}\n')
        print(f"{self.name}: Save Test results to {result_pth}")
    
    def load_model(self, model, run_name, is_gpu=True, is_current=False, get_ckpt=False):
        ckpt_name = 'best_model.pth' if not is_current else 'current_model.pth'
        ckpt_pth = os.path.join(self.root_dir, run_name, 'checkpoints', ckpt_name)
        ckpt = torch.load(ckpt_pth)
        model.load_state_dict(ckpt)
        if is_gpu:
            model = model.cuda()
        print(f"Load {ckpt_pth}")
        if get_ckpt: return model, ckpt
        return model
    
    def get_ckpt_pth(self):
        return os.path.join(self.root_dir, self.name, 'checkpoints', 'best_model.pth')

    def save_attack_counter(self, counter):
        path = os.path.join(self.attack_pth, 'attack_metrics.pkl')
        with open(path, 'wb') as f:
            pickle.dump(counter, f)
        
    def save_attack_visual(self, images, adv_images, attack_name):
        len_visual = min(images.shape[0], 8)
        output = list()
        for i in range(len_visual):
            output.append(normalize(images[i]))
            output.append(normalize(adv_images[i]))
        temp_visual_path = os.path.join(self.root_pth, \
            f'{attack_name}_adervsarial_temp.png')
        torchvision.utils.save_image(
            output, temp_visual_path, nrow=int(4)
        )

    def save_gradients(self, images, bp, guided_bp, pertubations):
        bp = bp.detach().cpu().numpy()
        guided_bp = guided_bp.detach().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        for i in range(images.shape[0]):
            save_path = os.path.join(self.attack_pth, f'{str(i)}_grad.png')
            bp_out = norm_01(bp[i])
            guided_bp_out = norm_01(guided_bp[i])
            pertubations_i = [norm_pertubation(item[i]) for item in pertubations]
            save = np.concatenate([normalize(images[i]).detach().cpu().numpy(), \
                bp_out, guided_bp_out] + pertubations_i, axis=-1) * 255
            save = Image.fromarray(save.astype(np.uint8).transpose(1,2,0))
            save.save(save_path)
    
    def print_features(self, features):
        # print #3 features
        num_features = len(features)
        for i in range(6):
            results = list()
            len_feature = features[0][i][3].shape[0]
            for k in range(min(len_feature, 100)):
                for j in range(num_features):
                    results.append(clip_01(features[j][i][3][k]))
            feature_pth = os.path.join(self.attack_visual_pth, f'feature_{i}.png')
            torchvision.utils.save_image(
                results, 
                feature_pth, 
                nrow=int(num_features)
            )

    def show_explaination(self, images, adv_images, label, attack_name, victim_models):
        norm_img = normalize(images) * 255
        norm_adv = normalize(adv_images) * 255
        pertubations = norm_img - norm_adv
        print(f"{attack_name}: Max pertubation {pertubations.abs().max()}")

        for i in range(images.shape[0]):
            # Creat Dir for each image
            img_pth = os.path.join(self.attack_visual_pth, str(i))
            if not os.path.exists(img_pth): os.mkdir(img_pth)

            # Creat Dir for attack and victim models
            attack_pth = os.path.join(img_pth, attack_name)
            if not os.path.exists(attack_pth): os.mkdir(attack_pth)

            for victim_name, model in victim_models.items():
                model_pth = os.path.join(attack_pth, victim_name)
                if not os.path.exists(model_pth): os.mkdir(model_pth)

                img = images[i].unsqueeze(0)
                adv_img = adv_images[i].unsqueeze(0)

                result_raw = model(img)[0]
                result_adv = model(adv_img)[0]
                
                # Save prob.txt for each img and each model
                string = gen_results_strings(label[i], result_raw, result_adv)
                with open(os.path.join(model_pth, 'prob.txt'), 'w') as f:
                    f.write(string)

                # Save features maps
                out, features = model(img, get_features=True)
                out, features_adv = model(adv_img, get_features=True)
                
                if i == 3:
                    for i in range(5):
                        fea_i, fea_i_adv = features[i][0], features_adv[i][0]
                        feature_pth = os.path.join(model_pth, f'feature_{i}.png')
                        result = list()
                        for j in range(fea_i.shape[0]):
                            result.append(clip_01(fea_i[j]))
                            result.append(clip_01(fea_i_adv[j]))
                        torchvision.utils.save_image(
                            result, 
                            feature_pth, 
                            nrow=int(16)
                        )

                torchvision.utils.save_image(
                    [normalize(img[0]), normalize(adv_img[0])], 
                    os.path.join(model_pth, 'imgs.png'), 
                    nrow=int(2)
                )
        import ipdb; ipdb.set_trace()