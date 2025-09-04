import os,sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from networks.tools.snn.quant_tools import *
from timm.models.vision_transformer import _cfg
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from functools import partial
from networks.tools.init_tools import *
from networks.tools.snn.quant_patch import *

class Patch_Embedding(nn.Module):
    def __init__(self, in_dim, embed_dim,T,kernel_size):
        super().__init__()
    
        self.patch_cnn=Patch_Conv(in_dim,embed_dim,T,kernel_size)
        
    def forward(self, x):
        x_patch=self.patch_cnn(x)
        return x_patch



class SDSA(nn.Module):
    def __init__(self, in_dim, T, kernel_size):
        super().__init__()
        self.T = T
        self.scale = in_dim ** (-0.5)
        self.num_head = 4
        head_dim = in_dim // self.num_head

        self.conv_qkv = nn.ModuleList([
            Quant_ConvBn(in_dim, in_dim, T, kernel_size) 
            for _ in range(3)
        ])
        
        self.conv_attn = nn.ModuleList([
            Quant_DSConv(in_dim, in_dim, T, kernel_size) 
            for _ in range(4)
        ])
        
        self.conv = Quant_MLP(in_dim * 2, in_dim, T)
        self.mlp = Quant_MLP(in_dim, in_dim, T)
        self.norm = Quant_Norm(in_dim, T)

    def forward_decomposing(self, x):
        q, k, v = [conv(x) for conv in self.conv_qkv]
        q_a, q_i = q, 1.0 - q
        k_a, k_i = k, 1.0 - k
        return q_a, q_i, k_a, k_i, v

    def compute_attention(self, q, k, v, conv_layer):
        attn_weight = (q @ k.transpose(-2, -1)) * self.scale
        attn_output = attn_weight @ v
        T, B, num_head, head_dim, N = attn_output.shape
        C = num_head * head_dim
        H = W = int(N ** 0.5)
        attn_output = attn_output.permute(0, 1, 2, 4, 3).contiguous()
        attn_output = attn_output.view(T, B, C, H, W)
        return conv_layer(attn_output)

    def reshape_for_attention(self, tensors, T, B, num_head, head_dim, N):
        reshaped_tensors = []
        for tensor in tensors:
            reshaped_tensor = tensor.view(T, B, num_head, head_dim, N)
            reshaped_tensors.append(reshaped_tensor)
        return reshaped_tensors

    def forward_mm_qk(self, q_a, q_i, k_a, k_i, v):
        T, B, C, H, W = v.shape
        N = H * W
        head_dim = C // self.num_head
        
        tensors_to_reshape = [q_a, q_i, k_a, k_i, v]
        reshaped_tensors = self.reshape_for_attention(tensors_to_reshape, T, B, self.num_head, head_dim, N)
        
        q_a, q_i, k_a, k_i, v = reshaped_tensors

        attn_aa = self.compute_attention(q_a, k_a, v, self.conv_attn[0])
        attn_ii = self.compute_attention(q_i, k_i, v, self.conv_attn[1])
        attn_ai = self.compute_attention(q_a, k_i, v, self.conv_attn[2])
        attn_ia = self.compute_attention(q_i, k_a, v, self.conv_attn[3])
        
        return [attn_aa, attn_ii, attn_ai, attn_ia]

    def forward_out(self, attn_list, v):
        attn_aa, attn_ii, attn_ai, attn_ia = attn_list
        attn = torch.cat([attn_aa + attn_ii, attn_ai + attn_ia], dim=2)
        out = self.conv(attn) + v
        out = self.norm(out)
        out = self.mlp(out) + out
        return out

    def forward(self, x):
        q_a, q_i, k_a, k_i, v = self.forward_decomposing(x)
        attn_list = self.forward_mm_qk(q_a, q_i, k_a, k_i, v)
        out = self.forward_out(attn_list, v)
        return out
    


class SDSA_Stage(nn.Module):
    def __init__(self, in_dim,out_dim,T,kernel_size):
        super().__init__()
        
        self.patch=Patch_Embedding(in_dim, out_dim,T,kernel_size)
        self.sdsa=SDSA(out_dim,T,kernel_size)
        self.mask_conv=Quant_Mask_PredictionHead(out_dim,T, kernel_size)
        self.pool_conv=Patch_Pool(out_dim, out_dim,T)
        
    
    def forward(self, x):
        
        x_patch=self.patch(x)
        out=self.sdsa(x_patch)
        mask=self.mask_conv(out)
        out=self.pool_conv(out)
        return out,mask

class Decoder(nn.Module):
    def __init__(self,in_dim,out_dim,T,kernel_size):
        super().__init__()
        self.sc=Quant_Spike_Conv(in_dim,in_dim,T,kernel_size)
        self.norm=Quant_Norm(in_dim,T)
        self.up=Patch_UpSample(in_dim,out_dim,T,kernel_size)
    
    def forward(self, x):
        x=self.sc(x)+x
        x=self.norm(x)
        x=self.up(x)
        return x



class Spike_Net(nn.Module):
    def __init__(self, in_dim, embedding_dim, T, kernel_size):
        super().__init__()
        self.T = T
        self.spike_tensor = Quant_Spike_Tensor(T)
        
        self.stages = nn.ModuleList()
        stage_dims = [in_dim] + embedding_dim
        for i in range(len(embedding_dim)):
            self.stages.append(SDSA_Stage(stage_dims[i], stage_dims[i+1], T, kernel_size))
        
        self.decoders = nn.ModuleList()
        decoder_dims = embedding_dim.copy()
        for i in range(len(decoder_dims)-1, 0, -1):
            in_channels = decoder_dims[i]
            out_channels = decoder_dims[i-1]
            self.decoders.append(Decoder(in_channels, out_channels, T, kernel_size))
        
        self.decoders.append(Decoder(embedding_dim[0], embedding_dim[0], T, kernel_size))
        
        self.head = PredictionHead(embedding_dim[0], T)

    def forward(self, x, mix=False):
        x = self.spike_tensor(x)
        
        stage_outputs = []
        masks = []
        
        for stage in self.stages:
            x, mask = stage(x)
            stage_outputs.append(x)
            masks.append(mask)
        
        decoder_input = stage_outputs[-1]
        
        for i in range(len(self.decoders)):
            decoder = self.decoders[i]
            decoded = decoder(decoder_input)
            
            if i < len(stage_outputs) - 1:
                skip_connection = stage_outputs[-2-i]
                decoder_input = decoded + skip_connection
            else:
                decoder_input = decoded
        
        out = self.head(decoder_input)
        
        if mix:
            return out, masks[0], masks[1], masks[2], masks[3]
        else:
            return out
        

class Spike_T_Net(nn.Module):
    def __init__(self, in_dim,embedding_dim,T,kernel_size,init_type,bn_init_type):
        super().__init__()
        self.spike_net=Spike_Net(in_dim,embedding_dim,T,kernel_size)
        self.spike_net.apply(lambda m: extended_init_weights(m, init_type, bn_init_type))
        
    def forward(self, x,mix=False):
        if mix:
            out_a,mask1_a,mask2_a,mask3_a,mask4_a=self.spike_net(x,mix)
            return out_a,mask1_a,mask2_a,mask3_a,mask4_a
        else:
            out_a=self.spike_net(x)
            return out_a
