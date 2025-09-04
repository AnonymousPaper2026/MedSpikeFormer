
import torch
import os
from utils.micro import *
from networks.tools.init_tools import *
from networks.ann_seg.ann_model import *

class AnnNetHandler:
    def __init__(self, device, checkpoint_path):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None

    def _create_model(self, in_channels, embedding_dim, kernel_size):
        return Ann_Net(in_channels, embedding_dim, kernel_size)

    def _check_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

    def _load_weights(self):
        self._check_checkpoint()
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=False)

    def _freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self, in_channels, embedding_dim, kernel_size, load_weights,freeze):
        self.model = self._create_model(in_channels, embedding_dim, kernel_size)
        self.model.apply(lambda m: extended_init_weights(m, init_type, bn_init_type))
        self.model = self.model.to(self.device)
        if load_weights==RESUME_Y:
            self._load_weights()
        if freeze==FREEZE_Y:
            self._freeze_params()
        return self.model

