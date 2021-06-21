import torch
from torchadver.attacker.iterative_gradient_attack import *

# FGSM_Linf = FGM_LInf

# BIM_L2 = I_FGM_L2
# I_FGSM_Linf = I_FGM_LInf

# M_BIM_L2 = MI_FGM_L2
# MI_FGSM_Linf = MI_FGM_LInf

# D_M_BIM_L2 = M_DI_FGM_L2
# DI_MI_FGSM_Linf = M_DI_FGM_LInf

# PGD_L2 = PGD_L2
# PGD_Linf = PGD_LInf



from advertorch.attacks import LinfBasicIterativeAttack, \
    LinfPGDAttack, LinfMomentumIterativeAttack, L2BasicIterativeAttack,\
        L2PGDAttack, CarliniWagnerL2Attack, ElasticNetL1Attack,\
            L2MomentumIterativeAttack

FGSM_L2 = L2BasicIterativeAttack
FGSM_Linf = LinfBasicIterativeAttack

I_FGSM_L2 = L2BasicIterativeAttack
I_FGSM_Linf = LinfBasicIterativeAttack

MI_FGSM_L2 = L2MomentumIterativeAttack
MI_FGSM_Linf = LinfMomentumIterativeAttack

# D_M_BIM_L2 = M_DI_FGM_L2
# DI_MI_FGSM_Linf = M_DI_FGM_LInf

PGD_L2 = L2PGDAttack
PGD_Linf = LinfPGDAttack

Noise_L2 = L2PGDAttack
Noise_Linf = LinfPGDAttack

CW_Linf = LinfBasicIterativeAttack
CW_L2 = CarliniWagnerL2Attack

EAD_L1 = CarliniWagnerL2Attack

image_net_mean = [0.5, 0.5, 0.5]
image_net_std = [0.5, 0.5, 0.5]

torch_mean = torch.Tensor(image_net_mean).view([3,1,1])
torch_std = torch.Tensor(image_net_std).view([3,1,1])

if torch.cuda.is_available():
    torch_std_cuda = torch_std.cuda()
    torch_mean_cuda = torch_mean.cuda()
