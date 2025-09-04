from utils.micro import *

args={
"datasets": Kvasir,
"gpu": 1,
"batchsize": 2,
"imagesize": SIZE128,
"log": "log",
"seed": 41,
"epoch": 400,
"in_channels": 3,
"embedding_dim": [32, 64, 128, 256],
"time": 4,
"lr": 0.0001,
"ti": 0,
"esp": 200,
"checkpoint_pth": "/home/xyq1/MyFolder/Net_Seg_tj/log/ADSA_Kvasir/checkpoint.pth",
"kernel_size": KERNEL3,
"freeze":FREEZE_Y,
"freeze_s":FREEZE_N,
"a_s": A_S_A,
"align": [0,0,0,1],
"opt":OPT_ADAM,
"lr":LR_RLROP,
"patience":30,
"resume":RESUME_N,
"resume_s":RESUME_N,
"model_type_s":SPIKE_MODEL,
"model_type_a":ANN_MODEL,
"weight_init_mode":INIT_KAIMING,
"bn_init_mode":BN_INIT_DEFAULT
}


