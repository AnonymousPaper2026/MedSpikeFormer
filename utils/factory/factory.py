import torch
import os
from utils.micro import RESUME_Y, FREEZE_Y, SPIKE_MODEL, ANN_MODEL
from networks.ann_seg.ann_model import Ann_Net
from networks.spike_seg.spike_model import Spike_T_Net

class ModelHandler:
    def __init__(self, device, checkpoint_path):
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.model = None

    def _create_model(self, model_type, *model_params):
        if model_type == SPIKE_MODEL:
            in_dim, embedding_dim, T, kernel_size, weigt_mode,bn_mode = model_params
            return Spike_T_Net(in_dim, embedding_dim, T, kernel_size,weigt_mode,bn_mode)
        elif model_type == ANN_MODEL:
            in_channels, embedding_dim, kernel_size, weigt_mode,bn_mode = model_params
            return Ann_Net(in_channels, embedding_dim, kernel_size, weigt_mode,bn_mode)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _check_checkpoint(self):
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

    def _load_weights(self):
        if self.checkpoint_path:
            self._check_checkpoint()
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)

    def _freeze_params(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def get_model(self, model_type, *model_params, load_weights, freeze):
        self.model = self._create_model(model_type, *model_params)
        self.model = self.model.to(self.device)
        if load_weights == RESUME_Y:
            self._load_weights()
        if freeze == FREEZE_Y:
            self._freeze_params()
        return self.model

def build_model(*kwargs):
    model_type = kwargs[0]
    handler = ModelHandler(kwargs[1], kwargs[2])
    
    if model_type == SPIKE_MODEL:
        return handler.get_model(
            model_type,
            kwargs[3], kwargs[4], kwargs[5], kwargs[6],kwargs[7],kwargs[8],
            load_weights=kwargs[9],
            freeze=kwargs[10]
        )
    elif model_type == ANN_MODEL:
        return handler.get_model(
            model_type,
            kwargs[3], kwargs[4], kwargs[5], kwargs[6],kwargs[7],
            load_weights=kwargs[8],
            freeze=kwargs[9]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")