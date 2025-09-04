import torch.nn as nn
import torch.nn.init as init
import torch
from utils.micro import *



def extended_init_weights(m, init_type, bn_init_type, **kwargs):
    if isinstance(m, nn.Conv2d):
        if init_type == INIT_KAIMING:
            mode = kwargs.get('mode', 'fan_out')
            nonlinearity = kwargs.get('nonlinearity', 'relu')
            init.kaiming_normal_(m.weight, mode=mode, nonlinearity=nonlinearity)
        elif init_type == INIT_XAVIER:
            gain = kwargs.get('gain', 1.0)
            init.xavier_normal_(m.weight, gain=gain)
        elif init_type == INIT_NORMAL:
            mean = kwargs.get('mean', 0.0)
            std = kwargs.get('std', 0.01)
            init.normal_(m.weight, mean=mean, std=std)
        elif init_type == INIT_UNIFORM:
            a = kwargs.get('a', -0.1)
            b = kwargs.get('b', 0.1)
            init.uniform_(m.weight, a=a, b=b)
        elif init_type == INIT_ORTHOGONAL:
            gain = kwargs.get('gain', 1.0)
            init.orthogonal_(m.weight, gain=gain)
        elif init_type == INIT_SPARSE:
            sparsity = kwargs.get('sparsity', 0.1)
            std = kwargs.get('std', 0.01)
            init.sparse_(m.weight, sparsity=sparsity, std=std)
        elif init_type == INIT_TRUNC_NORMAL:
            mean = kwargs.get('mean', 0.0)
            std = kwargs.get('std', 0.02)
            a = kwargs.get('a', -2.0)
            b = kwargs.get('b', 2.0)
            if hasattr(init, 'trunc_normal_'):
                init.trunc_normal_(m.weight, mean=mean, std=std, a=a, b=b)
            else:
                init.normal_(m.weight, mean=mean, std=std)
                with torch.no_grad():
                    m.weight.clamp_(min=a, max=b)
        elif init_type == INIT_ZEROS:
            init.zeros_(m.weight)
        elif init_type == INIT_ONES:
            init.ones_(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if bn_init_type == BN_INIT_DEFAULT:
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif bn_init_type == BN_INIT_ZEROS:
            init.constant_(m.weight, 0)
            init.constant_(m.bias, 0)
        elif bn_init_type == BN_INIT_SMALL:
            init.constant_(m.weight, 0.1)
            init.constant_(m.bias, 0)
        elif bn_init_type == BN_INIT_RANDOM:
            init.normal_(m.weight, mean=1.0, std=0.02)
            init.constant_(m.bias, 0)
        elif bn_init_type == BN_INIT_UNIFORM:
            init.uniform_(m.weight, a=0.5, b=1.5)
            init.constant_(m.bias, 0)


