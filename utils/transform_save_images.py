import os
import numpy as np
import albumentations as A
import torch
import torch.nn as nn
from utils.utils import calculate_metrics
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
import torch.nn.functional as F
from spikingjelly.clock_driven import functional

import matplotlib.pyplot as plt
from datetime import datetime

def transform_save_images(model, loader, loss_fn, device,epoch,name):
    model.eval()

    with torch.no_grad():
        for i, ((x), (y1, y2)) in enumerate(tqdm(loader, desc="Evaluation", total=len(loader))):
            x = x.to(device, dtype=torch.float32)
            y1 = y1.to(device, dtype=torch.float32)
            y2 = y2.to(device, dtype=torch.float32)
            functional.reset_net(model)
            mask_pred= model(x,mix=False)
            mask_pred_pic = mask_pred.cpu().squeeze().numpy()
            mask_pred_pic[mask_pred_pic>0.5]=1
            mask_pred_pic[mask_pred_pic<=0.5]=0
            plt.imshow(mask_pred_pic, cmap='gray')
            plt.axis('off')
            save_dir = f"Test/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = f"epoch{epoch}_iter{i}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0) 
            plt.close()
