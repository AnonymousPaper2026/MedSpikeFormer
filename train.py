import os
import time
import datetime
import torch
from utils.utils import print_and_save, epoch_time, my_seeding
from utils.metrics import DiceBCELoss, KLL2Loss
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
from utils.factory.factory import *
import argparse
from utils.utils import calculate_params_flops
from dataset.loader import get_loader
from utils.transform_save_images import transform_save_images
from networks.spike_seg.spike_model import Spike_T_Net
from networks.ann_seg.ann_create import AnnNetHandler
from networks.spike_seg.spike_create import SpikeNetHandler
from utils.run_train_vail import Trainer
import json
from utils.micro import *

def load_config(config_path=None):
    from config.config import args
    return args

class SegTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args["gpu"]}')
        self.setup_directories()
        self.setup_data_loaders()
        self.setup_models()
        self.setup_optimizer_scheduler()
        self.setup_loss_functions()
        self.setup_trainer()
        self.initialize_training_variables()
        
    def setup_directories(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{self.args['datasets']}_{current_time}"
        self.save_dir = os.path.join(self.args['log'], folder_name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.train_log_path = os.path.join(self.save_dir, "train_log.txt")
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pth")
        
        train_log = open(self.train_log_path, "w")
        train_log.write("\n")
        train_log.close()
        
        datetime_object = str(datetime.datetime.now())
        self.print_and_save_log(datetime_object)
        self.print_and_save_log("")
        
        hyperparameters_str = f"Image Size: {self.args['imagesize']}\nBatch Size: {self.args['batchsize']}\nLR: {self.args['lr']}\nEpochs: {self.args['epoch']}\n"
        hyperparameters_str += f"Early Stopping Patience: {self.args['esp']}\n"
        hyperparameters_str += f"Seed: {self.args['seed']}\n"
        self.print_and_save_log(hyperparameters_str)

    def setup_data_loaders(self):
        self.train_loader, self.valid_loader = get_loader(
            self.args['datasets'], 
            self.args['batchsize'], 
            self.args['imagesize'], 
            self.train_log_path
        )
    
    def setup_models(self):
        self.model = build_model(
            self.args['model_type_s'],
            self.device,
            None,
            self.args['in_channels'],
            self.args['embedding_dim'],
            self.args['time'],
            self.args['kernel_size'],
            self.args['weight_init_mode'],
            self.args['bn_init_mode'],
            self.args['resume_s'],
            self.args['freeze_s']
        )
        
        self.ann_model = build_model(
            self.args['model_type_a'],
            self.device,
            self.args['checkpoint_pth'],
            self.args['in_channels'],
            self.args['embedding_dim'],
            self.args['kernel_size'],
            self.args['weight_init_mode'],
            self.args['bn_init_mode'],
            self.args['resume_s'],
            self.args['freeze_s']
        )
        
        calculate_params_flops(self.model, self.args['imagesize'], self.device, self.train_log_path)
    
    def setup_optimizer_scheduler(self):
        param_groups = [{'params': self.model.parameters(), 'lr': self.args['lr']}]
        if self.args["opt"] == OPT_ADAM:
            self.optimizer = torch.optim.Adam(param_groups)
        
        if self.args["lr"] == LR_RLROP:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 'min', patience=self.args["patience"], verbose=True
            )
    
    def setup_loss_functions(self):
        self.loss_fn = DiceBCELoss()
        self.loss_kll = KLL2Loss()
        loss_name = "BCE Dice Loss"
        data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
        self.print_and_save_log(data_str)
        
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        data_str = f"Number of parameters: {num_params / 1000000}M\n"
        self.print_and_save_log(data_str)
    
    def setup_trainer(self):
        self.trainer = Trainer(
            self.model, 
            self.ann_model, 
            self.train_loader, 
            self.valid_loader, 
            self.optimizer, 
            self.loss_fn, 
            self.loss_kll, 
            self.device, 
            self.args
        )
    
    def initialize_training_variables(self):
        self.best_valid_metrics = 0.0
        self.early_stopping_count = 0
    
    def print_and_save_log(self, message):
        print_and_save(self.train_log_path, message)
    
    def run_training(self):
        for epoch in range(self.args['epoch']):
            start_time = time.time()
            self.trainer.train(epoch)
            valid_loss, valid_metrics = self.trainer.evaluate(epoch)
            
            if self.args["lr"] == LR_RLROP:
                self.scheduler.step(valid_loss)
            
            self.check_and_save_best_model(valid_metrics, epoch)
            
            if self.early_stopping_count == self.args['esp']:
                data_str = f"Early stopping: validation loss stops improving from last {self.args['esp']} continously.\n"
                self.print_and_save_log(data_str)
                break
            
            end_time = time.time()
            self.log_epoch_time(start_time, end_time, epoch, valid_loss, valid_metrics)
    
    def check_and_save_best_model(self, valid_metrics, epoch):
        if valid_metrics[0] > self.best_valid_metrics:
            data_str = f"Valid mIoU improved from {self.best_valid_metrics:2.4f} to {valid_metrics[0]:2.4f}. Saving checkpoint: {self.checkpoint_path}"
            self.print_and_save_log(data_str)
            
            self.best_valid_metrics = valid_metrics[0]
            torch.save(self.model.state_dict(), self.checkpoint_path)
            self.early_stopping_count = 0
        elif valid_metrics[0] < self.best_valid_metrics:
            self.early_stopping_count += 1
    
    def log_epoch_time(self, start_time, end_time, epoch, valid_loss, valid_metrics):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        data_str = f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - miou:{valid_metrics[0]},dsc:{valid_metrics[1]},acc:{valid_metrics[2]},sen:{valid_metrics[3]},spe:{valid_metrics[4]},pre:{valid_metrics[5]},rec:{valid_metrics[6]},fb:{valid_metrics[7]},em:{valid_metrics[8]}\n"
        self.print_and_save_log(data_str)

if __name__ == "__main__":
    args = load_config()
    my_seeding(args['seed'])
    
    seg_trainer = SegTrainer(args)
    seg_trainer.run_training()