# A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks

by Qingsong Yao, Zecheng He, Yi Lin, Kai Ma, Yefeng Zheng and S. Kevin Zhou

The source code of paper "A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks" early accepted by MICCAI 2021 (AC Track 1/18).

## Abstract

Deep neural networks for medical images are extremely vulnerable to adversarial examples (AEs), which poses security concerns on clinical decision-making. Recent findings have shown that existing medical AEs are easy to detect in feature space. To better understand this phenomenon, we thoroughly investigate the characteristic of traditional medical AEs in feature space. Specifically, we first perform a stress test to reveal the vulnerability of medical images and compare them to natural images. Then, we theoretically prove that the existing adversarial attacks manipulate the prediction by continuously optimizing the vulnerable representations in a fixed direction, leading to outlier representations in feature space. Interestingly, we find this vulnerability is a double-edged sword that can be exploited to help hide AEs in the feature space. We propose a novel hierarchical feature constraint (HFC) as an add-on to existing white-box attacks, which encourages hiding the adversarial representation in the normal feature distribution. We evaluate the proposed method on two public medical image datasets, namely {Fundoscopy} and {Chest X-Ray}. Experimental results demonstrate the superiority of our HFC as it bypasses an array of state-of-the-art adversarial medical AEs detector more efficiently than competing adaptive attacks.

## Effects and command of each python file:

- train.py : Train CNN classifier 

  parameters: -a arch -n name (save ckpts to runs/dataset/name)

  save ckpts & print performances

  ```bash
  mkdir runs_APTOS
  python train.py -a resnet50 -n resnet50 --dataset aptos -b 64 --lr 0.0003
  ```

- gen_base_adv.py : Generate adversarial examples 

  -n name = -a arch

  attacks: I_FGSM FGSM MI_FGSM PGD CW (after setting confidence score)

  perturbation-type: Linf L2  perturbation-size: 1 2 3 4 5 .. / 256

  ```bash
  python gen_base_adv.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1
  ```

- extract_features.py: Train adversarail detectors and save as pickle files.

  -n name = -a arch

  attack: The specified adversarail attack (generated and stored by gen_base_adv.py)

  need to generate Noise_Linf_2 firstly:
  ```bash
  python gen_base_adv.py -a resnet50 --dataset APTOS --attack Noise_Linf_2
  ```

  ```bash
  python extract_features.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1
  ```

- get_GMM.py: train gmm models for each layer

  -n name = -a arch; -y number of components

  ```bash
  python get_GMM.py -a resnet50 --dataset APTOS -y 2
  ```

- gmm_attack.py : attack with HFC plug-in term, generate AEs with HFC and load detectors trained by extract_features.py

  --detector: which detectors are used to detect adversarail examples

  ```bash
  python gmm_attack.py -a resnet50 --dataset APTOS --attack I_FGSM_Linf_1 --detector I_FGSM_Linf_1
  ```

## Other python files:

- eval.py : test the performances of HFC (need to generate adversarial examples of HFC first)
- saver.py utils.py network.py vgg.py resnet.py : utils

## Citation
If our work is useful to you, please cite our paper as:

```latex
@artical{qsyao2020landmarkattack,
  title={Miss the Point: Targeted Adversarial Attack on Multiple Landmark Detection},
  author={Qingsong Yao, Zecheng He, Yi Lin, Kai Ma, Yefeng Zheng, and S. Kevin Zhou},
  booktitle={A Hierarchical Feature Constraint to Camouflage Medical Adversarial Attacks},
  year={2021}
}
```
