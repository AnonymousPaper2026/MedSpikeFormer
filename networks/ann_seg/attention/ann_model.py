import os,sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from networks.tools.ann.ann_patch import *
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from networks.tools.init_tools import *
from networks.tools.ann.ann_tools import *
from functools import partial


from networks.tools.init_tools import *

class Patch_Embedding(nn.Module):
    def __init__(self, in_dim, embed_dim, kernel_size):
        super().__init__()
    
        self.patch_cnn=Ann_Patch_Conv(in_dim,embed_dim, kernel_size)
        
    def forward(self, x):
        x_patch=self.patch_cnn(x)
        return x_patch



class ADSA(nn.Module):
    def __init__(self, in_dim, kernel_size):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** (-0.5)
        self.num_head = 4
        self.head_dim = in_dim // self.num_head

        self.conv_qkv = nn.ModuleList([
            Ann_ConvBn(in_dim, in_dim, kernel_size) for _ in range(3)
        ])
        
        self.conv_attn_list = nn.ModuleList([
            Ann_DSConv(in_dim, in_dim, kernel_size) for _ in range(4)
        ])
        
        self.conv = Ann_MLP(in_dim * 2, in_dim, kernel_size)
        self.mlp = Ann_MLP(in_dim, in_dim, kernel_size)
        self.norm = Ann_Norm(in_dim)

    def forward_decomposing(self, x):
        q, k, v = [conv(x) for conv in self.conv_qkv]
        q_a, q_i = q, 1.0 - q
        k_a, k_i = k, 1.0 - k
        return q_a, q_i, k_a, k_i, v

    def _reshape_for_attention(self, tensor, B, N):
        return tensor.view(B, self.num_head, self.head_dim, N)

    def _compute_attention_pair(self, q, k, v, conv_layer):
        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        attn_output = attn_weight @ v
        return attn_output

    def forward_mm_qk(self, q_a, q_i, k_a, k_i, v):
        B, C, H, W = v.shape
        N = H * W

        q_a_rs = self._reshape_for_attention(q_a, B, N)
        q_i_rs = self._reshape_for_attention(q_i, B, N)
        k_a_rs = self._reshape_for_attention(k_a, B, N)
        k_i_rs = self._reshape_for_attention(k_i, B, N)
        v_rs = self._reshape_for_attention(v, B, N)

        attn_aa = self._compute_attention_pair(q_a_rs, k_a_rs, v_rs, self.conv_attn_list[0])
        attn_ii = self._compute_attention_pair(q_i_rs, k_i_rs, v_rs, self.conv_attn_list[1])
        attn_ai = self._compute_attention_pair(q_a_rs, k_i_rs, v_rs, self.conv_attn_list[2])
        attn_ia = self._compute_attention_pair(q_i_rs, k_a_rs, v_rs, self.conv_attn_list[3])

        attn_aa = self.conv_attn_list[0](attn_aa.view(B, C, H, W))
        attn_ii = self.conv_attn_list[1](attn_ii.view(B, C, H, W))
        attn_ai = self.conv_attn_list[2](attn_ai.view(B, C, H, W))
        attn_ia = self.conv_attn_list[3](attn_ia.view(B, C, H, W))
        
        return attn_aa, attn_ii, attn_ai, attn_ia

    def forward_out(self, attn_aa, attn_ii, attn_ai, attn_ia, v):
        attn = torch.cat([attn_aa + attn_ii, attn_ai + attn_ia], dim=1)
        out = self.conv(attn) + v
        out = self.norm(out)
        out = self.mlp(out) + out
        return out

    def forward(self, x):
        q_a, q_i, k_a, k_i, v = self.forward_decomposing(x)
        attn_list = self.forward_mm_qk(q_a, q_i, k_a, k_i, v)
        out = self.forward_out(*attn_list, v)
        return out




class ADSA_Stage(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        
        self.patch=Patch_Embedding(in_dim, out_dim,kernel_size)
        self.adsa=ADSA(out_dim,kernel_size)
        self.mask_conv=Ann_Mask_PredictionHead(out_dim)
        self.pool_conv=Ann_Patch_Pool(out_dim, out_dim)
        
    
    def forward(self, x):
        
        x_patch=self.patch(x)
        out=self.adsa(x_patch)
        mask=self.mask_conv(out)
        out=self.pool_conv(out)
        return out,mask

class Decoder(nn.Module):
    def __init__(self,in_dim,out_dim,kernel_size):
        super().__init__()
        self.sc=Ann_Conv(in_dim,kernel_size)
        self.norm=Ann_Norm(in_dim)
        self.up=Ann_Patch_UpSample(in_dim,out_dim,kernel_size)
    
    def forward(self, x):
        x=self.sc(x)+x
        x=self.norm(x)
        x=self.up(x)
        return x



class Ann_Net(nn.Module):
    def __init__(self, in_dim: int, embedding_dim: list, kernel_size: int,init_type,bn_init_type):
        super().__init__()
        
        self.stages = nn.ModuleList()
        stage_dims = [in_dim] + embedding_dim
        for i in range(len(embedding_dim)):
            self.stages.append(ADSA_Stage(stage_dims[i], stage_dims[i+1], kernel_size))
        
        self.decoders = nn.ModuleList()
        decoder_dims = embedding_dim[::-1] 
        for i in range(len(decoder_dims) - 1):
            self.decoders.append(Decoder(decoder_dims[i], decoder_dims[i+1], kernel_size))
        self.decoders.append(Decoder(embedding_dim[0], embedding_dim[0], kernel_size))
        
        self.head = Ann_PredictionHead(embedding_dim[0])
        self.apply(lambda m: extended_init_weights(m, init_type, bn_init_type))

    def forward(self, x: torch.Tensor, pre: bool = False) -> torch.Tensor:
        
        stage_outputs = [] 
        masks = []
        for stage in self.stages:
            x, mask = stage(x)
            stage_outputs.append(x)
            masks.append(mask)
        
        if pre:
            return masks
        
        decoder_input = stage_outputs[-1]
        
        for i, decoder in enumerate(self.decoders):
            decoded = decoder(decoder_input)
            if i < len(stage_outputs) - 1:
                skip_connection = stage_outputs[-2-i]
                decoder_input = decoded + skip_connection 
            else:
                decoder_input = decoded
        
        out = self.head(decoder_input)
        return out

