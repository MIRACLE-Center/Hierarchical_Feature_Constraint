import SimpleITK as sitk
from SimpleITK.SimpleITK import Normalize
import numpy as np
import csv
import os
import traceback
import random
from PIL import Image
from tqdm import tqdm 
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
import cv2
import pickle
import random
import logging
from scipy.interpolate import interpn
import copy
import torchvision.transforms as transforms

from sklearn.model_selection import KFold


'''
rand_pairs: used to control attack mode

targeted attack: inverse attack use all of the datasets, inverse the lable (binary datasets)
specific: train and break detector for specific class
train_single_class : get image from the targeted class
'''
def interpolate_3d(arr):
    # 定义缩小比例
    scale_factor = 0.5

    # 计算插值后的新形状
    new_shape = tuple(int(dim * scale_factor) for dim in arr.shape)

    # 创建用于插值的坐标网格
    x_coords = np.linspace(0, arr.shape[0]-1, arr.shape[0])
    y_coords = np.linspace(0, arr.shape[1]-1, arr.shape[1])
    z_coords = np.linspace(0, arr.shape[2]-1, arr.shape[2])

    # 计算插值后的坐标
    x_new = np.linspace(0, arr.shape[0]-1, new_shape[0])
    y_new = np.linspace(0, arr.shape[1]-1, new_shape[1])
    z_new = np.linspace(0, arr.shape[2]-1, new_shape[2])
    coords = np.meshgrid(x_new, y_new, z_new, indexing='ij')
    coords = np.stack(coords, axis=-1)

    # 进行线性插值
    interp_arr = interpn((x_coords, y_coords, z_coords), arr, coords)

    return interp_arr

def generate_spatial_bounding_box(img, channel_indexes=None, margin=0):
    assert isinstance(margin, int), "margin must be int type."
    temp = np.zeros_like(img[0]).astype(np.float)
    temp = img.mean(axis=0)
    temp = cv2.medianBlur(temp.astype(np.uint8),9)
    nonzero_idx = np.nonzero(temp > 5)
    box_start = list()
    box_end = list()
    for i in range(temp.ndim):
        assert len(nonzero_idx[i]) > 0, f"did not find nonzero index at spatial dim {i}"
        box_start.append(max(0, np.min(nonzero_idx[i]) - margin))
        box_end.append(min(temp.shape[i], np.max(nonzero_idx[i]) + margin + 1))
    return box_start, box_end

def unpack(array):
    lenth = array['img'].shape[0]
    ret = [[id, array['label'][id]] for id in range(lenth)]
    return ret

def load_image_to_numpy_array(file_path):
    try:
        # load the data once
        itk_img = sitk.ReadImage(file_path)
        img_array = sitk.GetArrayFromImage(itk_img)  # indexes are z,y,x (notice the ordering)
        # img_array = img_array.transpose(0, 2, 1)  # take care on the sequence of axis of v_center ,transfer to x,y,z
        img_array = itensity_normalize_one_volume(img_array)
        img_array = np.expand_dims(img_array, axis=0)

    except Exception as e:
        print(" process images %s error..." % str(os.path.basename(file_path)))
        # print(Exception, ":", e)
        traceback.print_exc()
    img_array = interpolate_3d(img_array[0])
    return img_array[np.newaxis, :]


def Normalization(volume):
    ######### max-min (0-1) #############
    max = np.max(volume)
    min = np.min(volume)
    if max == min:
        if max == 0:
            return volume
        else:
            return volume/max
    else:
        volume = (volume - min) / (max - min)  # float cannot apply the compute,or array error will occur
        return volume


def itensity_normalize_one_volume(volume):
    volume = volume.astype(np.float64)
    pixels = volume[volume > 0].astype(np.float64)
    max = pixels.max()
    min = pixels.min()
    out = volume
    out[volume > 0] = (pixels - min + 0.5)/(max - min)
    # volume[volume == 0] = 0
    return out

def ImageToTensor(img):
    return torch.from_numpy(np.array(img, dtype=np.float32))

def MaskToTensor(img):
    return torch.from_numpy(np.array(img, dtype=np.int32)).long()

class Brain(Dataset):
    def __init__(self, root_pth='./datasets/Brain/',\
        mode='train', num_fold=0, targeted=False, rand_pairs=False, target_class=0, arch='resnet'):
        self.num_classes = 4
        self.num_classes_selected = 4
        self.training = mode == 'train'
        self.cell_size = (30, 128, 128)

        self.root_path = root_pth
        if mode == 'train':
            label_csv = os.path.join(root_pth, '5fold_label', 'train_5_1500.csv')
        else:
            label_csv = os.path.join(root_pth, '5fold_label', 'test_5_1500.csv')
        f = open(label_csv, 'r')
        f_csv = csv.reader(f)

        self.image_list = list()
        for raw_id, row in enumerate(f_csv):
            self.image_list.append((os.path.join(root_pth, 'brain_CT', row[0]), \
                int(float(row[2]))-1))

        if mode == 'train':
            pass
        elif mode == 'test':
            if rand_pairs is not None:
                bingo = np.load(os.path.join(f'/home1/qsyao/Code_HFC/runs_Brain/resnet3d/correct_predicts.npy'))
                len_raw = len(self.image_list)
                self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
                print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
            pass
        else:
            bingo = np.load(os.path.join(f'/home1/qsyao/Code_HFC/runs_Brain/resnet3d/correct_predicts.npy'))
            len_raw = len(self.image_list)
            self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
            print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
            len_image_list = len(self.image_list)
            split_fold = 0.310
            split_flag = int(len_image_list * (1 - split_fold))
            if mode == 'adv_train':
                self.image_list = self.image_list[:split_flag]
            elif mode == 'adv_test':
                self.image_list = self.image_list[split_flag:]
            else:
                raise NotImplementedError   
        
        # Prepare to get one target sample from train set
        self.rand_pairs = rand_pairs
        self.class_dict = dict()
        for i in range(self.num_classes):
            self.class_dict[i] = list()
        for item in self.image_list:
            label = item[1]
            self.class_dict[label].append([item[0], label])
        # for i in range(self.num_classes):
        #     print(f'Class {i} num images {len(self.class_dict[i])}')

        if rand_pairs == 'train_single_class':
            self.image_list = self.class_dict[target_class]
            print(f'Mode: Train single class, num images: {len(self.image_list)} class id {target_class}')
            mode = 'test'
            self.target_image_patch = list()
            for item in self.class_dict[target_class]:
                img_pth = item[0]
                img_data = self.load_process(img_pth)
                self.target_image_patch.append(img_data.unsqueeze(0))
            self.target_image_patch = torch.cat(self.target_image_patch)

        pickle_pth = os.path.join(root_pth, 'target_class.pkl')
        self.target_class_list = {id_class:list() for id_class in range(self.num_classes)}
        for key in self.target_class_list.keys():
            self.target_class_list[key] = list(range(self.num_classes))
            self.target_class_list[key].remove(key)

        if not os.path.exists(pickle_pth) or True:
            print("Set target class to {}".format(pickle_pth))
            with open(pickle_pth, 'wb') as f:
                self.attack_list = list()
                self.class_selected_list = {_:[] for _ in range(self.num_classes)}
                for item in self.image_list:
                    id_class = item[1]
                    rand_class = random.choice(self.target_class_list[id_class])
                    self.attack_list.append([item[0], rand_class])
                    self.class_selected_list[rand_class].append(item[0])
                pickle.dump([self.attack_list, self.class_selected_list], f)
        with open(pickle_pth, 'rb') as f:
            logging.info("Load attack list and target class list from {}".format(pickle_pth))
            self.attack_list, self.class_selected_list = pickle.load(f)

        # Get class_spcific
        if rand_pairs == 'specific':
            assert(mode != 'train')
            self.target_class = target_class
            # self.temp_list = [[item, target_class] for item in self.class_selected_list[target_class]]
            self.temp_list = []
            for item in self.image_list:
                if item[0] in self.class_selected_list[target_class]:
                    self.temp_list.append([item[0], target_class])
            self.image_list = self.temp_list
        
        if rand_pairs == 'targeted_attack':
            assert(mode != 'train')
            self.image_list = self.attack_list
        
        self.rand_pairs = rand_pairs
    
        self.targeted = targeted
        self.target_class = target_class

    def __len__(self):
        return len(self.image_list)

    def load_process(self, img_path):
        img = load_image_to_numpy_array(img_path)
        # img = Normalization(img)
        img = ImageToTensor(img)
        return img

    def __getitem__(self, index):
        if self.rand_pairs == 'train_single_class':
            img_data = self.target_image_patch[index]
            id_target = np.random.randint(0, len(self.target_image_patch))
            target_data = self.target_image_patch[id_target]
            return img_data, self.target_class, target_data
        item = self.image_list[index]

        img_pth = item[0]
        img_data = self.load_process(img_pth)

        label = item[1]
        return img_data, torch.tensor(label).long()

class CXR(Dataset):
    def __init__(self, root_pth='./datasets/cxr/',\
        mode='train', num_fold=0, targeted=False, rand_pairs=None, target_class=0, arch='vgg16'):
        self.num_classes = 2
        self.num_classes_selected = 2
        targeted = False

        self.image_list = list()
        self.plan_pkl_pth = os.path.join(root_pth, 'plans.pkl')
        
        if not os.path.exists(self.plan_pkl_pth):
            logging.warning("Generate 8-2 split to {}".format(self.plan_pkl_pth))
            train_folder_h = os.path.join(root_pth, 'train', 'NORMAL')
            train_healthy = [[os.path.join(train_folder_h, item), 0] for item in os.listdir(train_folder_h)]
            test_folder_h = os.path.join(root_pth, 'test', 'NORMAL')
            test_healthy = [[os.path.join(test_folder_h, item), 0] for item in os.listdir(test_folder_h)]
            train_folder_p = os.path.join(root_pth, 'train', 'PNEUMONIA')
            train_pneumonia = [[os.path.join(train_folder_p, item), 1] for item in os.listdir(train_folder_p)]
            test_folder_p = os.path.join(root_pth, 'test', 'PNEUMONIA')
            test_pneumonia = [[os.path.join(test_folder_p, item), 1] for item in os.listdir(test_folder_p)]
            train_healthy.extend(train_pneumonia)
            test_healthy.extend(test_pneumonia)
            random.shuffle(train_healthy)
            random.shuffle(test_healthy)
            test_healthy.extend(train_healthy[-550:])
            train_healthy = train_healthy[:-550]
            self.plan = dict()
            self.plan['test'] = test_healthy
            self.plan['train'] = train_healthy
            with open(os.path.join(root_pth, 'plans.pkl'), 'wb') as f:
                pickle.dump(self.plan, f)
        
        with open(self.plan_pkl_pth, 'rb') as f:
            self.plan = pickle.load(f)
        # # import ipdb; ipdb.set_trace()        
        self.img_dir_pth = os.path.join(root_pth, 'images')

        transform_list = list()
        transform_list.append(transforms.Resize(280))
        if mode == 'train':
            transform_list.append(transforms.RandomCrop(256))
            transform_list.append(transforms.RandomHorizontalFlip())
        else:
            transform_list.append(transforms.CenterCrop(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0, 0, 0],\
                                    std=[1, 1, 1]))
        self.transform = transforms.Compose(transform_list)

        # # resize 
        # images = os.listdir(self.img_dir_pth)
        # for item in tqdm(images):
        #     image = Image.open(os.path.join(self.img_dir_pth, item)).resize([330, 330])
        #     image.save(os.path.join(self.img_dir_pth, item))
        
        if mode == 'train':
            self.image_list = self.plan['train']
        elif mode == 'test':
            self.image_list = self.plan['test']
            if rand_pairs is not None:
                len_raw = len(self.image_list)
                bingo = np.load(os.path.join(f'./runs_CXR/resnet50/correct_predicts.npy'))
                self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
                print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
        else:
            self.image_list = self.plan['test']
            len_raw = len(self.image_list)
            bingo = np.load(os.path.join(f'./runs_CXR/resnet50/correct_predicts.npy'))
            self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
            print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
            len_image_list = len(self.image_list)
            split_fold = 0.3
            split_flag = int(len_image_list * (1 - split_fold))
            if mode == 'adv_train':
                self.image_list = self.image_list[:split_flag]
            elif mode == 'adv_test':
                self.image_list = self.image_list[split_flag:]
            else:
                raise NotImplementedError   
        
        # Prepare to get one target sample from train set
        self.rand_pairs = rand_pairs
        self.class_dict = dict()
        for i in range(self.num_classes):
            self.class_dict[i] = list()
        for item in self.plan['train']:
            label = item[1]
            self.class_dict[label].append([item[0], label])
        if rand_pairs == 'train_single_class':
            self.image_list = self.class_dict[target_class]
            mode = 'test'
            self.target_image_patch = list()
            for item in self.class_dict[target_class]:
                img_pth = item[0]
                img_data = self.transform(Image.open(img_pth).convert('RGB'))
                self.target_image_patch.append(img_data.unsqueeze(0))
            self.target_image_patch = torch.cat(self.target_image_patch)

        pickle_pth = os.path.join(root_pth, 'target_class.pkl')
        self.target_class_list = {id_class:list() for id_class in range(self.num_classes)}
        for key in self.target_class_list.keys():
            self.target_class_list[key] = list(range(self.num_classes))
            self.target_class_list[key].remove(key)

        if not os.path.exists(pickle_pth) or False:
            assert(mode == 'test' and rand_pairs == 'targeted_attack')
            print("Set target class to {}".format(pickle_pth))
            with open(pickle_pth, 'wb') as f:
                self.attack_list = list()
                self.class_selected_list = {_:[] for _ in range(self.num_classes)}
                for item in self.image_list:
                    id_class = int(int(item[1]) > 0)
                    rand_class = random.choice(self.target_class_list[id_class])
                    self.attack_list.append([item[0], rand_class])
                    self.class_selected_list[rand_class].append(item[0])
                pickle.dump([self.attack_list, self.class_selected_list], f)

        with open(pickle_pth, 'rb') as f:
            logging.info("Load attack list and target class list from {}".format(pickle_pth))
            self.attack_list, self.class_selected_list = pickle.load(f)

        # Get class_spcific
        if rand_pairs == 'specific':
            assert(mode != 'train')
            self.target_class = target_class
            # self.temp_list = [[item, target_class] for item in self.class_selected_list[target_class]]
            self.temp_list = []
            for item in self.image_list:
                if item[0] in self.class_selected_list[target_class]:
                    self.temp_list.append([item[0], target_class])
            self.image_list = self.temp_list
        
        if rand_pairs == 'targeted_attack':
            assert(mode != 'train')
            self.image_list = self.attack_list
        
        self.rand_pairs = rand_pairs
    
        self.targeted = targeted
        self.target_class = target_class

    def get_target_one(self, id_class):
        item = random.choice(self.class_dict[id_class])
        img_pth = item[0]
        img_data = self.transform(Image.open(img_pth).convert('RGB'))
        return img_data
    
    def get_random_feature(self, id_class, num_samples=100):
        random.shuffle(self.class_dict[id_class])
        target_list = self.class_dict[id_class][:num_samples]
        target_images = [self.transform(Image.open(target_list[i][0]).convert('RGB')).cuda().unsqueeze(0)\
                 for i in range(num_samples)]
        return torch.cat(target_images)
    
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        if self.rand_pairs == 'train_single_class':
            img_data = self.target_image_patch[index]
            id_target = np.random.randint(0, len(self.target_image_patch))
            target_data = self.target_image_patch[id_target]
            return img_data, self.target_class, target_data
        item = self.image_list[index]

        img_pth = item[0]
        img_data = self.transform(Image.open(img_pth).convert('RGB'))

        label = item[1]
        
        return img_data, label

class APTOS(Dataset):
    def __init__(self, root_pth='./datasets/APTOS/',\
        mode='train', num_fold=0, targeted=False, rand_pairs=None, target_class=0, arch='resnet50'):

        self.num_classes = 2
        self.num_classes_selected = 2
        # Fixed : targeted attack:
        # rand_pairs control mode
        targeted = False

        transform_list = list()
        transform_list.append(transforms.Resize(280))
        if mode == 'train':
            transform_list.append(transforms.RandomCrop(256))
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomVerticalFlip())
        else:
            transform_list.append(transforms.CenterCrop(256))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0, 0, 0],\
                                    std=[1, 1, 1]))
        self.transform = transforms.Compose(transform_list)

        ## Resize the large images
        self.raw_dir_pth = os.path.join(root_pth, 'train_images')
        self.img_dir_pth = os.path.join(root_pth, 'processed_images')
        if not os.path.exists(self.img_dir_pth):
            os.mkdir(self.img_dir_pth)
            files = os.listdir(self.raw_dir_pth)
            for name in tqdm(files, desc='Preprocess Resize'):
                image = Image.open(os.path.join(self.raw_dir_pth, name))
                array = np.array(image).transpose(2,0,1)
                bbox_start, bbox_end = generate_spatial_bounding_box(array)
                array = array[:, bbox_start[0]:bbox_end[0], bbox_start[1]:bbox_end[1]]
                image = Image.fromarray(array.transpose(1,2,0)).resize([256,256])
                image.save(os.path.join(self.img_dir_pth, name))

        self.image_list = list()

        self.train_csv_pth = os.path.join(root_pth, 'train.csv')
        
        with open(self.train_csv_pth, 'r') as csv_file:
            reader = csv.reader(csv_file)

            for item in reader:
                if reader.line_num == 1:
                    continue

                self.image_list.append([item[0], item[1]])

        # Creat 2-8 split
        import random
        pickle_pth = os.path.join(root_pth, 'plan.pkl')
        if not os.path.exists(pickle_pth):
            logging.warning("Generate 8-2 split to {}".format(pickle_pth))
            with open(pickle_pth, 'wb') as f:
                split_5_folds = list()
                random.shuffle(self.image_list)
                self.image_list = np.array(self.image_list)
                kl_fold = KFold()
                splits = kl_fold.split(self.image_list)
                for train_index, test_index in splits:
                    k_split = dict()  
                    train_list = self.image_list[train_index].tolist()
                    test_list = self.image_list[test_index].tolist()
                    k_split['train_list'] = train_list
                    k_split['test_list'] = test_list
                    split_5_folds.append(k_split)
                pickle.dump(split_5_folds, f)

        with open(pickle_pth, 'rb') as f:
            logging.info("Load 8-2 split from {}".format(pickle_pth))
            split_file = pickle.load(f)[num_fold]

        if mode == 'train':
            self.image_list = split_file['train_list']
        elif mode == 'test':
            self.image_list = split_file['test_list']
            if rand_pairs is not None:
                len_raw = len(self.image_list)
                bingo = np.load(os.path.join(f'./runs_APTOS/resnet50/correct_predicts.npy'))
                self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
                print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
        else:
            self.image_list = split_file['test_list']
            len_raw = len(self.image_list)
            bingo = np.load(os.path.join(f'./runs_APTOS/resnet50/correct_predicts.npy'))
            self.image_list = [self.image_list[i] for i in range(bingo.shape[0]) if bingo[i]]
            print(f"Drop {len_raw - len(self.image_list)} samples from test set from {len_raw}")
            len_image_list = len(self.image_list)
            split_fold = 0.3
            split_flag = int(len_image_list * (1 - split_fold))
            if mode == 'adv_train':
                self.image_list = self.image_list[:split_flag]
            elif mode == 'adv_test':
                self.image_list = self.image_list[split_flag:]
            else:
                raise NotImplementedError
        # import ipdb; ipdb.set_trace()
        # Only for try_random_pairs experiments
        self.rand_pairs = rand_pairs
        self.class_dict = dict()
        # 2 classes
        for i in range(2):
            self.class_dict[i] = list()
        for item in split_file['train_list']:
            label = int(int(item[1]) > 0)
            self.class_dict[label].append([item[0], label])
        if rand_pairs == 'train_single_class':
            self.image_list = self.class_dict[target_class]
            mode = 'test'
            self.target_image_patch = list()
            for item in self.class_dict[target_class]:
                img_pth = os.path.join(self.img_dir_pth, item[0]+'.png')
                img_data = self.transform(Image.open(img_pth).convert('RGB'))
                self.target_image_patch.append(img_data.unsqueeze(0))
            self.target_image_patch = torch.cat(self.target_image_patch)
        
        pickle_pth = os.path.join(root_pth, 'target_class.pkl')
        self.target_class_list = {id_class:list() for id_class in range(self.num_classes)}
        for key in self.target_class_list.keys():
            self.target_class_list[key] = list(range(self.num_classes))
            self.target_class_list[key].remove(key)

        if not os.path.exists(pickle_pth) or False:
            # assert(mode == 'test' and rand_pairs == 'targeted_attack')
            print("Set target class to {}".format(pickle_pth))
            with open(pickle_pth, 'wb') as f:
                self.attack_list = list()
                self.class_selected_list = {_:[] for _ in range(self.num_classes)}
                for item in self.image_list:
                    id_class = int(int(item[1]) > 0)
                    rand_class = random.choice(self.target_class_list[id_class])
                    self.attack_list.append([item[0], rand_class])
                    self.class_selected_list[rand_class].append(item[0])
                pickle.dump([self.attack_list, self.class_selected_list], f)

        with open(pickle_pth, 'rb') as f:
            logging.info("Load attack list and target class list from {}".format(pickle_pth))
            self.attack_list, self.class_selected_list = pickle.load(f)

        # Get class_spcific
        if rand_pairs == 'specific':
            assert(mode != 'train')
            self.target_class = target_class
            # self.temp_list = [[item, target_class] for item in self.class_selected_list[target_class]]
            self.temp_list = []
            for item in self.image_list:
                if item[0] in self.class_selected_list[target_class]:
                    self.temp_list.append([item[0], target_class])
            self.image_list = self.temp_list
        
        if rand_pairs == 'targeted_attack':
            assert(mode != 'train')
            self.image_list = self.attack_list
        
        self.rand_pairs = rand_pairs
        self.target_class = target_class
    
        self.targeted = targeted

    def get_target_one(self, id_class):
        item = random.choice(self.class_dict[id_class])
        img_pth = os.path.join(self.img_dir_pth, item[0]+'.png')
        img_data = self.transform(Image.open(img_pth).convert('RGB'))
        return img_data
    
    def get_target_batch(self, id_class, size):
        res = list()
        for i in range(size):
            item = random.choice(self.class_dict[id_class])
            img_pth = os.path.join(self.img_dir_pth, item[0]+'.png')
            img_data = self.transform(Image.open(img_pth).convert('RGB'))
            res.append(img_data.unsqueeze(0))
        res = torch.cat(res)
        return res

    def get_random_feature(self, id_class, num_samples=100):
        random.shuffle(self.class_dict[id_class])
        target_list = self.class_dict[id_class][:num_samples]
        target_images = [self.transform(Image.open(os.path.join(\
            self.img_dir_pth, target_list[i][0]+'.png')).convert('RGB')).cuda().unsqueeze(0)\
                 for i in range(num_samples)]
        return torch.cat(target_images)

    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        if self.rand_pairs == 'train_single_class':
            img_data = self.target_image_patch[index]
            id_target = np.random.randint(0, len(self.target_image_patch))
            target_data = self.target_image_patch[id_target]
            return img_data, self.target_class, target_data
        item = self.image_list[index]

        img_pth = os.path.join(self.img_dir_pth, item[0]+'.png')
        img_data = self.transform(Image.open(img_pth).convert('RGB'))

        label = int(int(item[1]) > 0)
        
        return img_data, label

def get_dataloader(dataset='APTOS',\
    mode='train', batch_size=8, num_workers=16, num_fold=0, \
        targeted=False, rand_pairs=None, target_class=0, arch='resnet50'):
    arch = arch.split('_')[0]
    # Now: Default : targeted = True
    num_workers = 24
    is_train = mode == 'train'
    if dataset == 'APTOS':
        dataset = APTOS(mode=mode, num_fold=num_fold, targeted=targeted, \
            rand_pairs=rand_pairs, target_class=target_class, arch=arch)
    elif dataset == 'Brain':
        dataset = Brain(mode=mode, num_fold=num_fold, targeted=targeted, \
            rand_pairs=rand_pairs, target_class=target_class)
    elif dataset == 'CXR':
        dataset = CXR(mode=mode, num_fold=num_fold, targeted=targeted, \
            rand_pairs=rand_pairs, target_class=target_class, arch=arch)
    # elif dataset == 'Cifar':
    #     dataset = Cifar(mode=mode, num_fold=num_fold, targeted=targeted, \
    #         rand_pairs=rand_pairs, target_class=target_class, arch='resnet50')
    dataset.__getitem__(0)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
        shuffle=is_train
    )
    return loader

if __name__ == "__main__":
    # # Debug Dataset
    # test_set = APTOS(xxx)
    # for i in range(len(test_set.__len__())):
    #     test = test_set.__getitem__(i)

    # Debug Loader
    loader = get_dataloader(dataset='Brain', mode='test')
    test = loader.dataset.__getitem__(0)
    # import ipdb; ipdb.set_trace()
    # for item in tqdm(range(loader.dataset.__len__())):
    #     test = loader.dataset.__getitem__(item)
    for img, label in tqdm(loader, desc='Test Dataset'):
        # import ipdb; ipdb.set_trace()
        pass
    