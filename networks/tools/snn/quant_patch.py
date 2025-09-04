import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


class Quant_ConvBn_skip(nn.Module):
    def __init__(self, in_dim,out_dim,T,kernel_szie):
        super().__init__()
        self.T=T
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_szie,padding=kernel_szie//2,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.bn2=nn.BatchNorm2d(out_dim)
        
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x_skip=x
        x=x.view(self.T*B,C,H,W)
        x=self.conv(x)
        x=self.bn(x)
        x=x.view(self.T,B,C,H,W)
        x=self.act(x)
        x=x+x_skip
        x=x.view(self.T*B,C,H,W)
        x=self.bn2(x)
        x=x.view(self.T,B,C,H,W)
        return x
    

    
class ConvBn(nn.Module):
    def __init__(self, in_dim,out_dim,T,kernel_size):
        super().__init__()
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2)
        self.bn=nn.BatchNorm2d(out_dim)
        self.T=T
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.conv(x)
        x=self.bn(x)
        Tb,C,H,W=x.shape
        B=Tb//self.T
        x=x.view(self.T,B,C,H,W)
        return x


    

class Patch_Conv(nn.Module):
    def __init__(self, in_dim, embed_dim,T,kernel_size):
        super().__init__()
    
        self.patch_cnn=ConvBn(in_dim,embed_dim,T,kernel_size)
        self.conv=Quant_ConvBn_skip(embed_dim,embed_dim,T,kernel_size)
        
    
    def forward(self, x):
        x_patch=self.patch_cnn(x)
        x=self.conv(x_patch)
        return x_patch



    
class ConvBnPool(nn.Module):
    def __init__(self, in_dim,out_dim,T):
        super().__init__()
        self.conv=nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0)
        self.bn=nn.BatchNorm2d(out_dim)
        self.T=T
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.conv(x)
        x=self.bn(x)
        Tb,C,H,W=x.shape
        B=Tb//self.T
        x=x.view(self.T,B,C,H,W)
        return x
    
    
class Patch_Pool(nn.Module):
    def __init__(self, in_dim, embed_dim,T):
        super().__init__()
    
        self.pool_conv=ConvBnPool(in_dim,embed_dim,T)
        
    def forward(self, x):
        x_pool = self.pool_conv(x)
        return x_pool



class Decoder(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.sc=Quant_Spike_Conv(in_dim)
        self.up=nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_dim)
        )
    
    def forward(self, x):
        x=self.sc(x)+x
        x=self.up(x)
        return x


    
class Patch_UpSample(nn.Module):
    def __init__(self, in_dim, out_dim,T,kernel_size):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv=ConvBn(in_dim,out_dim,T,kernel_size)
        self.T=T
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(B*self.T,C,H,W)
        x = self.up(x)
        Tb,C,H,W=x.shape
        B=Tb//self.T
        x=x.view(self.T,B,C,H,W)
        x=self.conv(x)
        return x



class PredictionHead(nn.Module):
    def __init__(self, in_dim,T):
        super().__init__()
        self.merge=nn.Sequential(
            nn.Conv2d(in_dim,in_dim,kernel_size=3,padding=1),
            nn.BatchNorm2d(in_dim),
        )
        self.conv=nn.Conv2d(in_dim,1,kernel_size=1)
        self.sig=nn.Sigmoid()
        self.T=T
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.mean(dim=0).view(B,C,H,W)
        x=self.merge(x)
        x=self.conv(x)
        x=self.sig(x)
        return x






class DSConvBn(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.pconv=nn.Conv2d(out_dim,out_dim,kernel_size=1)
        self.bn2=nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.pconv(x)
        x=self.bn2(x)
        return x
    
    
class Quant_Mask_PredictionHead(nn.Module):
    def __init__(self, in_dim,T, kernel_size):
        super().__init__()
        self.T=T
        self.merge=DSConvBn(in_dim, in_dim, kernel_size)
        self.conv=nn.Conv2d(in_dim,1,kernel_size=1)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.mean(dim=0).view(B,C,H,W)
        x=self.merge(x)
        x=self.conv(x)
        x=self.sig(x)
        return x


