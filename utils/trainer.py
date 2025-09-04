import os
import sys
import time
import datetime
import torch
import argparse
from typing import Tuple, Dict, Optional

# 导入自定义模块（按功能分类，符合A类论文模块化规范）
from utils.utils import print_and_save, epoch_time, my_seeding, calculate_params_flops
from utils.metrics import DiceBCELoss, KLL2Loss
from dataset.loader import get_loader
from utils.factory import build_model
from utils.run_train_vail import Trainer  # 复用原有Trainer类（train_val.py不变）
from utils.micro import OPT_ADAM, LR_RLROP  # 统一管理常量


def parse_cli_args() -> argparse.Namespace:
    """
    解析命令行参数（A类论文必备：支持批量实验、参数快速调整，无需修改代码）
    Returns:
        argparse.Namespace: 命令行参数对象
    """
    parser = argparse.ArgumentParser(
        description="Training Pipeline for Spike-ANN Alignment Segmentation (CCF A-Style)"
    )
    # 核心实验配置
    parser.add_argument(
        "--config-path", 
        type=str, 
        default="config/config.json", 
        help="Path to experiment config file (default: config/config.json)"
    )
    parser.add_argument(
        "--gpu-id", 
        type=int, 
        default=None, 
        help="Override GPU ID in config (e.g., 0, 1; default: use config's 'gpu' value)"
    )
    # 实验增强参数（A类论文常见需求）
    parser.add_argument(
        "--resume", 
        action="store_true", 
        help="Whether to resume training from checkpoint (default: False)"
    )
    parser.add_argument(
        "--resume-path", 
        type=str, 
        default=None, 
        help="Path to resume checkpoint (required if --resume is True)"
    )
    return parser.parse_args()


def load_experiment_config(config_path: str, cli_args: argparse.Namespace) -> Dict:
    """
    加载实验配置（合并配置文件与命令行参数，保障参数可追溯）
    Args:
        config_path: 配置文件路径
        cli_args: 命令行参数对象
    Returns:
        Dict: 合并后的完整实验配置
    """
    # 从配置文件加载基础参数
    from config.config import args as config_args
    # 合并命令行参数（命令行参数优先级更高，用于快速调整）
    if cli_args.gpu_id is not None:
        config_args["gpu"] = cli_args.gpu_id
    config_args["resume_training"] = cli_args.resume
    config_args["resume_checkpoint_path"] = cli_args.resume_path
    config_args["config_source_path"] = config_path  # 记录配置来源，便于复现
    
    # 补充A类论文必要的默认参数（避免KeyError）
    default_params = {
        "num_workers": 4,  # 数据加载多线程（提升效率）
        "weight_decay": 1e-5,  # 权重衰减（正则化，防止过拟合）
        "cudnn_benchmark": True,  # 加速CUDA推理
        "cudnn_deterministic": False  # 平衡速度与确定性
    }
    for key, val in default_params.items():
        if key not in config_args:
            config_args[key] = val
    
    return config_args


def init_experiment_directory(args: Dict) -> Tuple[str, str, str]:
    """
    初始化实验目录（日志、 checkpoint 、配置备份，A类论文要求完整追溯）
    Args:
        args: 实验配置
    Returns:
        Tuple: (保存根目录, 日志文件路径, 最佳模型路径)
    """
    # 目录命名规则：数据集_模型类型_时间（便于区分多组实验）
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args['datasets']}_{args['model_type_s']}_{current_time}"
    save_root_dir = os.path.join(args["log"], experiment_name)
    os.makedirs(save_root_dir, exist_ok=True)
    
    # 定义核心文件路径
    log_path = os.path.join(save_root_dir, "train_log.txt")
    best_model_path = os.path.join(save_root_dir, "best_model.pth")
    config_backup_path = os.path.join(save_root_dir, "experiment_config_backup.json")
    
    # 备份配置文件（A类论文关键：避免实验后参数丢失）
    import json
    with open(config_backup_path, "w", encoding="utf-8") as f:
        json.dump(args, f, indent=4, ensure_ascii=False)
    
    # 初始化日志文件
    with open(log_path, "w") as f:
        f.write(f"Experiment Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Experiment Name: {experiment_name}\n")
        f.write(f"Config Backup Path: {config_backup_path}\n\n")
    
    return save_root_dir, log_path, best_model_path


def log_hyperparameters(args: Dict, log_path: str) -> None:
    """
    记录超参数（A类论文要求：完整打印所有关键参数，便于复现和对比）
    Args:
        args: 实验配置
        log_path: 日志文件路径
    """
    hyperparam_sections = [
        # 1. 基础环境参数
        ("Environment", [
            f"GPU ID: {args['gpu']}",
            f"CUDA Benchmark: {args['cudnn_benchmark']}",
            f"CUDA Deterministic: {args['cudnn_deterministic']}",
            f"Random Seed: {args['seed']}"
        ]),
        # 2. 数据相关参数
        ("Data", [
            f"Dataset Name: {args['datasets']}",
            f"Image Size: {args['imagesize']}x{args['imagesize']}",
            f"Batch Size: {args['batchsize']}",
            f"Data Loader Workers: {args['num_workers']}",
            f"Input Channels: {args['in_channels']}"
        ]),
        # 3. 模型相关参数
        ("Model", [
            f"Spike Model Type: {args['model_type_s']}",
            f"ANN Model Type: {args['model_type_a']}",
            f"Embedding Dimension: {args['embedding_dim']}",
            f"Spike Time Steps: {args['time'] if 'time' in args else 'N/A'}",
            f"Kernel Size: {args['kernel_size']}",
            f"Freeze Spike Model: {args['freeze_s']}",
            f"Resume Spike Model: {args['resume_s']}"
        ]),
        # 4. 训练相关参数
        ("Training", [
            f"Total Epochs: {args['epoch']}",
            f"Initial LR: {args['lr']}",
            f"Weight Decay: {args['weight_decay']}",
            f"Optimizer: {OPT_ADAM}",
            f"LR Scheduler: {LR_RLROP} (Patience: {args['patience']})",
            f"Main Loss: DiceBCELoss",
            f"Alignment Loss: KLL2Loss",
            f"Early Stopping Patience: {args['esp']}",
            f"Resume Training: {args['resume_training']} (Path: {args['resume_checkpoint_path']})"
        ])
    ]
    
    # 写入日志
    print_and_save(log_path, "="*60)
    print_and_save(log_path, "Hyperparameters (Full Experiment Config)")
    print_and_save(log_path, "="*60)
    for section_name, params in hyperparam_sections:
        print_and_save(log_path, f"\n【{section_name}】")
        for param in params:
            print_and_save(log_path, f"  - {param}")
    print_and_save(log_path, "\n" + "="*60)


class SegmentationTrainer:
    """
    分割任务训练器类（A类论文风格：封装核心逻辑，便于复用和扩展）
    职责：模型初始化、优化器配置、训练循环控制、结果记录
    """
    def __init__(self, args: Dict, log_path: str, best_model_path: str):
        self.args = args
        self.log_path = log_path
        self.best_model_path = best_model_path
        
        # 初始化设备（A类论文要求显式设备配置）
        self.device = self._init_device()
        # 初始化数据加载器
        self.train_loader, self.valid_loader = self._init_data_loaders()
        # 初始化模型（Spike + ANN）
        self.spike_model, self.ann_model = self._init_models()
        # 初始化优化器、调度器、损失函数
        self.optimizer, self.scheduler, self.loss_fn, self.loss_kll = self._init_training_components()
        # 初始化训练核心类（复用原有train_val.py的Trainer）
        self.core_trainer = self._init_core_trainer()
        
        # 训练状态变量（可复现性：记录关键状态）
        self.best_val_miou = 0.0  # 核心评价指标（分割任务常用mIoU）
        self.early_stopping_count = 0
        self.start_epoch = 0  # 支持续训
        
        # 打印实验初始化信息
        self._log_init_info()

    def _init_device(self) -> torch.device:
        """初始化计算设备（CPU/CUDA）"""
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args['gpu']}")
            print_and_save(self.log_path, f"Using CUDA device: cuda:{self.args['gpu']}")
        else:
            device = torch.device("cpu")
            print_and_save(self.log_path, "CUDA not available, using CPU (WARNING: slow training)")
        # 配置CUDA参数（平衡速度与确定性）
        torch.backends.cudnn.benchmark = self.args["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = self.args["cudnn_deterministic"]
        return device

    def _init_data_loaders(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """初始化训练/验证数据加载器（A类论文要求显式数据信息）"""
        print_and_save(self.log_path, f"\nLoading dataset: {self.args['datasets']}")
        train_loader, valid_loader = get_loader(
            dataset_name=self.args['datasets'],
            batch_size=self.args['batchsize'],
            image_size=self.args['imagesize'],
            log_path=self.log_path,
            num_workers=self.args['num_workers']  # 补充多线程加载
        )
        # 记录数据加载器信息
        print_and_save(
            self.log_path,
            f"Data Loader Info: Train batches={len(train_loader)}, Valid batches={len(valid_loader)}"
        )
        return train_loader, valid_loader

    def _init_models(self) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """初始化Spike模型和ANN模型（工厂模式，支持多模型切换）"""
        print_and_save(self.log_path, f"\nInitializing models: Spike={self.args['model_type_s']}, ANN={self.args['model_type_a']}")
        
        # 1. 初始化Spike模型
        spike_model = build_model(
            model_type=self.args['model_type_s'],
            device=self.device,
            checkpoint_path=None,  # Spike模型默认从头训练
            in_channels=self.args['in_channels'],
            embedding_dim=self.args['embedding_dim'],
            time_steps=self.args.get('time', 8),  # Spike模型特有参数
            kernel_size=self.args['kernel_size'],
            resume=self.args['resume_s'],
            freeze=self.args['freeze_s']
        )
        
        # 2. 初始化ANN模型（用于对齐训练）
        ann_model = build_model(
            model_type=self.args['model_type_a'],
            device=self.device,
            checkpoint_path=self.args['checkpoint_pth'],  # ANN模型加载预训练权重
            in_channels=self.args['in_channels'],
            embedding_dim=self.args['embedding_dim'],
            kernel_size=self.args['kernel_size'],  # ANN模型无time_steps参数
            resume=self.args['resume_s'],
            freeze=self.args['freeze_s']
        )
        
        # 3. 计算模型复杂度（A类论文必备：参数量、FLOPs）
        calculate_params_flops(spike_model, self.args['imagesize'], self.device, self.log_path)
        trainable_params = sum(p.numel() for p in spike_model.parameters() if p.requires_grad)
        print_and_save(
            self.log_path,
            f"Spike Model Trainable Params: {trainable_params / 1e6:.2f}M"
        )
        
        # 4. 支持续训（加载历史模型状态）
        if self.args['resume_training'] and self.args['resume_checkpoint_path'] is not None:
            self._load_resume_checkpoint(spike_model)
        
        return spike_model, ann_model

    def _init_training_components(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.nn.Module, torch.nn.Module]:
        """初始化优化器、学习率调度器、损失函数（A类论文要求显式训练策略）"""
        # 1. 优化器（带权重衰减，防止过拟合）
        param_groups = [
            {
                'params': self.spike_model.parameters(),
                'lr': self.args['lr'],
                'weight_decay': self.args['weight_decay']  # 补充正则化
            }
        ]
        if self.args["opt"] == OPT_ADAM:
            optimizer = torch.optim.Adam(param_groups)
        else:
            raise ValueError(f"Unsupported optimizer: {self.args['opt']} (only ADAM supported now)")
        
        if self.args["lr_scheduler"] == LR_RLROP: 
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', patience=self.args["patience"], verbose=True, factor=0.5
            )
        else:
            raise ValueError(f"Unsupported LR scheduler: {self.args['lr_scheduler']} (only RLROP supported now)")
        
        loss_fn = DiceBCELoss()  
        loss_kll = KLL2Loss()    
        
        return optimizer, scheduler, loss_fn, loss_kll

    def _init_core_trainer(self) -> Trainer:
        return Trainer(
            model=self.spike_model,
            ann_model=self.ann_model,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            loss_kll=self.loss_kll,
            device=self.device,
            args=self.args
        )

    def _load_resume_checkpoint(self, model: torch.nn.Module) -> None:
        if not os.path.exists(self.args['resume_checkpoint_path']):
            raise FileNotFoundError(f"Resume checkpoint not found: {self.args['resume_checkpoint_path']}")
        
        checkpoint = torch.load(self.args['resume_checkpoint_path'], map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1  
        self.best_val_miou = checkpoint['best_val_miou']
        
        print_and_save(self.log_path, f"\nResumed from checkpoint: {self.args['resume_checkpoint_path']}")
        print_and_save(self.log_path, f"Resume Info: Start Epoch={self.start_epoch}, Best Val mIoU={self.best_val_miou:.4f}")

    def _log_init_info(self) -> None:
        print_and_save(self.log_path, "\n" + "="*60)
        print_and_save(self.log_path, "Experiment Initialization Completed")
        print_and_save(self.log_path, "="*60)

    def _log_epoch_result(self, epoch: int, epoch_mins: int, epoch_secs: int, val_loss: float, val_metrics: list) -> None:
        metrics_str = (
            f"miou:{val_metrics[0]:.4f}, "
            f"dsc:{val_metrics[1]:.4f}, "
            f"acc:{val_metrics[2]:.4f}, "
            f"sen:{val_metrics[3]:.4f}, "
            f"spe:{val_metrics[4]:.4f}, "
            f"pre:{val_metrics[5]:.4f}, "
            f"rec:{val_metrics[6]:.4f}, "
            f"fb:{val_metrics[7]:.4f}, "
            f"em:{val_metrics[8]:.4f}"
        )
        log_str = (
            f"Epoch: {epoch:02d} | Time: {epoch_mins}m {epoch_secs}s\n"
            f"\tValidation Loss: {val_loss:.4f}\n"
            f"\tValidation Metrics: {metrics_str}\n"
        )
        print_and_save(self.log_path, log_str)

    def run_training_loop(self) -> None:
        print_and_save(self.log_path, f"\nStarting Training Loop: Total Epochs={self.args['epoch']}, Start Epoch={self.start_epoch}")
        print_and_save(self.log_path, "="*60)

        for epoch in range(self.start_epoch, self.args['epoch']):
            start_time = time.time()
            self.core_trainer.train(epoch)
            
            val_loss, val_metrics = self.core_trainer.evaluate(epoch)  
            
            self.scheduler.step(val_loss)
            
            current_val_miou = val_metrics[0]
            if current_val_miou > self.best_val_miou:
                self.best_val_miou = current_val_miou
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.spike_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'best_val_miou': self.best_val_miou,
                    'config': self.args  
                }, self.best_model_path)
                print_and_save(
                    self.log_path,
                    f"[Epoch {epoch:02d}] Best Val mIoU Updated: {self.best_val_miou:.4f} (Saved to {self.best_model_path})"
                )
                self.early_stopping_count = 0 
            else:
                self.early_stopping_count += 1  
            
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            self._log_epoch_result(epoch, epoch_mins, epoch_secs, val_loss, val_metrics)
            
            if self.early_stopping_count >= self.args['esp']:
                print_and_save(
                    self.log_path,
                    f"\nEarly Stopping Triggered: No improvement for {self.args['esp']} consecutive epochs"
                )
                break
        
        print_and_save(self.log_path, "\n" + "="*60)
        print_and_save(self.log_path, "Training Completed")
        print_and_save(self.log_path, f"Final Best Val mIoU: {self.best_val_miou:.4f}")
        print_and_save(self.log_path, f"Best Model Path: {self.best_model_path}")
        print_and_save(self.log_path, f"Training End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_and_save(self.log_path, "="*60)


def main():
    cli_args = parse_cli_args()
    
    args = load_experiment_config(cli_args.config_path, cli_args)
    
    my_seeding(args['seed'])
    
    save_dir, log_path, best_model_path = init_experiment_directory(args)
    
    log_hyperparameters(args, log_path)
    
    trainer = SegmentationTrainer(args, log_path, best_model_path)
    trainer.run_training_loop()


if __name__ == "__main__":
    main()
