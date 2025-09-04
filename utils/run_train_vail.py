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
from utils.micro import *

import torch
import numpy as np
from tqdm import tqdm

class Trainer:
    def __init__(self, model, ann_model, train_loader, valid_loader, 
                 optimizer, loss_fn, loss_kll, device, args):
        self.model = model
        self.ann_model = ann_model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loss_kll = loss_kll
        self.device = device
        self.args = args
        self.metrics_keys = ['miou', 'dsc', 'acc', 'sen', 'spe', 'pre', 'rec', 'fb', 'em']
        
    def _initialize_metrics(self):
        return {
            'loss': 0.0,
            'miou': 0.0,
            'dsc': 0.0,
            'acc': 0.0,
            'sen': 0.0,
            'spe': 0.0,
            'pre': 0.0,
            'rec': 0.0,
            'fb': 0.0,
            'em': 0.0
        }

    def train(self, epoch):
        self.model.train()
        metrics = self._initialize_metrics()
        
        for i, (x, y1) in enumerate(tqdm(self.train_loader, desc="Training", total=len(self.train_loader))):
            x = x.to(self.device, dtype=torch.float32)
            y1 = y1.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()
            functional.reset_net(self.model)
            
            # 根据不同模式计算损失
            if self.args["a_s"] == A_S_S:
                out_s = self.model(x, mix=False)
                loss = self.loss_fn(out_s, y1)
            elif self.args["a_s"] == A_S_A:
                out_a = self.ann_model(x, pre=False)
                loss = self.loss_fn(out_a, y1)
            elif self.args["a_s"] == A_S_:
                pred_s = self.model(x, mix=True)
                out_s, mask1_s, mask2_s, mask3_s, mask4_s = pred_s
                mask_s = [mask1_s, mask2_s, mask3_s, mask4_s]
                loss_s = self.loss_fn(out_s, y1)
                
                with torch.no_grad():
                    pred_a = self.ann_model(x, pre=True)
                    mask1_a, mask2_a, mask3_a, mask4_a = pred_a
                    mask_a = [mask1_a, mask2_a, mask3_a, mask4_a]
                loss = loss_s
                for i in self.args['align']:
                    loss += self.loss_kll(mask_s[i], mask_a[i]) / len(self.args['align'])

            loss.backward()
            self.optimizer.step()
            metrics['loss'] += loss.item()

        metrics['loss'] /= len(self.train_loader)
        return metrics

    def evaluate(self, epoch):
        self.model.eval()
        metrics = self._initialize_metrics()
        
        with torch.no_grad():
            for i, (x, y1) in enumerate(tqdm(self.valid_loader, desc="Evaluation", total=len(self.valid_loader))):
                x = x.to(self.device, dtype=torch.float32)
                y1 = y1.to(self.device, dtype=torch.float32)
                
                functional.reset_net(self.model)
                mask_pred = self.model(x, mix=False)
                loss = self.loss_fn(mask_pred, y1)
                metrics['loss'] += loss.item()

                batch_metrics = {key: [] for key in self.metrics_keys}
                for yt, yp in zip(y1, mask_pred):
                    scores = calculate_metrics(yt, yp)
                    for idx, key in enumerate(self.metrics_keys):
                        batch_metrics[key].append(scores[idx])

                for key in self.metrics_keys:
                    metrics[key] += np.mean(batch_metrics[key])

        for key in metrics:
            metrics[key] /= len(self.valid_loader)

        return metrics['loss'], [metrics[key] for key in self.metrics_keys]
