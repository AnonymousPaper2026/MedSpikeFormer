import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


def kl_divergence_loss(O_s, O_a, epsilon=1e-8):
    """
    计算SDSA与ADSA特征间的KL散度损失
    
    参数:
        O_s (torch.Tensor): SDSA输出特征，形状(B, C, H, W)
        O_a (torch.Tensor): ADSA输出特征，形状(B, C, H, W)
        epsilon (float): 数值稳定项，防止除零错误
    
    返回:
        torch.Tensor: KL散度损失值
    """
    # 确保数值稳定性，将值限制在[epsilon, 1]范围内
    O_s = torch.clamp(O_s, epsilon, 1.0)
    O_a = torch.clamp(O_a, epsilon, 1.0)
    
    # 计算KL散度: O_s * log(O_s / O_a)
    kl_div = O_s * torch.log(O_s / O_a)
    
    # 对所有维度求平均 (批次、通道、空间维度)
    loss = torch.mean(kl_div)
    
    return loss

def l2_loss(O_s, O_a):
    """
    计算SDSA与ADSA特征间的L2损失
    
    参数:
        O_s (torch.Tensor): SDSA输出特征，形状(B, C, H, W)
        O_a (torch.Tensor): ADSA输出特征，形状(B, C, H, W)
    
    返回:
        torch.Tensor: L2损失值
    """
    # 计算平方差
    squared_diff = torch.pow(O_s - O_a, 2)
    
    # 对所有维度求平均 (批次、通道、空间维度)
    loss = torch.mean(squared_diff)
    
    return loss



class KLL2Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(KLL2Loss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        kl=kl_divergence_loss(inputs,targets)
        l2=l2_loss(inputs,targets)
        loss=kl+l2
        return loss

import numpy as np
from sklearn.metrics import confusion_matrix
def get_metrics(preds,gts):
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)
    y_pre = np.where(preds>=0.5, 1, 0)
    y_true = np.where(gts>=0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre,labels=[0, 1])
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 
    precision=float(TP)/float(TP+FP) if float(TP + FP) != 0 else 0
    recall=float(TP)/float(TP+FN) if float(TP + FN) != 0 else 0
    f_beta=(1+0.3)*(precision*recall)/(0.3*precision+recall) if float(0.3*precision+recall) != 0 else 0
    e_measure=1 - (1 / (1 + 1) * ((precision * recall) / (1 * precision + recall))) if float(1 * precision + recall) != 0 else 0
    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    log_info = f'miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, precision: {precision},\
                     recall: {recall}, f_beta: {f_beta}, e_measure: {e_measure},'
   
    return miou, f1_or_dsc, accuracy,sensitivity, specificity,precision,recall,f_beta,e_measure
    