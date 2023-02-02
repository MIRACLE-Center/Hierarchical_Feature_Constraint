# A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks

by Qingsong Yao, Zecheng He, Yi Lin, Kai Ma, Yefeng Zheng and S. Kevin Zhou

The source code of paper "A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks" early accepted by MICCAI 2021 (AC Track 1/18).

## Abstract

Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw on clinical decision making. It has been discovered that conventional medical AEs are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and rethink the reliability of reactive defenses, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that the conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs.
We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks.

## Download datsets
Download Fundoscopy (APTOS) or Chest X-ray (CXR) to ./datasets/
- APTOS: https://www.kaggle.com/c/aptos2019-blindness-detection
- Chest X-ray: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## To launch the HFC attack, please run each python file:

- train.py : Train CNN classifier 

  parameters: -a arch -n name (save ckpts to runs/dataset/name)

  save ckpts & print performances

  ```bash
  mkdir runs_APTOS
  mkdir runs_CXR
  ```

  ```bash
  python train.py -a resnet50 -n resnet50 --dataset aptos -b 64 --lr 0.0003
  ```

- gen_base_adv.py : Generate adversarial examples 

  -a name of architechture: resnet50 or vgg16

  attacks: I_FGSM FGSM MI_FGSM TI-FGSM DI-FGSM PGD CW

  perturbation-type: L_inf 
  
  perturbation-size: 1 2 3 4 5 .. / 256


  ```bash
  python gen_base_adv.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1 (DI_FGSM_Linf_1, etc)
  ```

- extract_features.py: Train adversarail detectors and save as pickle files.

  -a arch resnet50 or vgg16

  attack: The specified adversarail attack (generated and stored by gen_base_adv.py)

  Please generate Noise_Linf_2 firstly:
  ```bash
  python gen_base_adv.py -a resnet50 --dataset APTOS --attack Noise_Linf_2
  ```

  ```bash
  python extract_features.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1
  ```

- get_GMM.py: train gmm models for each layer

  -n name = -a arch; -y number of components

  ```bash
  python get_GMM.py -a resnet50 --dataset APTOS -y 64
  ```

- HFC_Attack.py : attack with HFC plug-in term, generate AEs with HFC and load detectors trained by extract_features.py

  --detector: the detectors trained by whitch attack are used to detect adversarail examples. The detectors containes KD, LID, MAHA, SVM, DNN, and BU.

  ```bash
  python HFC_Attack.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1 --detector I_FGSM_Linf_1
  ```

## Other python files:

- eval.py : test the performances of HFC (Please generate adversarial examples of HFC first.)
- saver.py utils.py network.py vgg.py resnet.py : utils

## For OOD detection:

  ```bash
  python test_ood.py --dataset APTOS
  ```

## Argments:

- --dataset: ATPOS and CXR for fundooscopy and Cest X-Ray, respectively
- --attack: PGD_Linf_1: PGD attack with perturbation budget Linf=1


## Citation
If our work is useful to you, please cite our paper as:

```latex
@inproceedings{yao2020hierarchical,
  title={A hierarchical feature constraint to camouflage medical adversarial attacks},
  author={Yao, Qingsong and He, Zecheng and Lin, Yi and Ma, Kai and Zheng, Yefeng and Zhou, S Kevin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={36--47},
  year={2021},
  organization={Springer}
}
```
