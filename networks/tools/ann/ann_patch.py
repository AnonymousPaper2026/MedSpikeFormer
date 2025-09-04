import torch
import torch.nn as nn


class Ann_ConvBn_skip(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size=3):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        x_skip=x
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        x=x+x_skip
        return x
    

    
class ConvBn(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2)
        self.bn=nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        return x
    

    

class Ann_Patch_Conv(nn.Module):
    def __init__(self, in_dim, embed_dim, kernel_size):
        super().__init__()
    
        self.patch_cnn=ConvBn(in_dim,embed_dim,kernel_size)
        self.conv=Ann_ConvBn_skip(embed_dim,embed_dim)
    
    def forward(self, x):
        x_patch=self.patch_cnn(x)
        x=self.conv(x_patch)
        return x_patch



    
class ConvBnPool(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()
        self.conv=nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2, padding=0)
        self.bn=nn.BatchNorm2d(out_dim)
        
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        return x
    
    
class Ann_Patch_Pool(nn.Module):
    def __init__(self, in_dim=3, embed_dim=3):
        super().__init__()
    
        self.pool_conv=ConvBnPool(in_dim,embed_dim)
        
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


    
class Ann_Patch_UpSample(nn.Module):
    def __init__(self, in_dim, out_dim,kernel_size):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv=ConvBn(in_dim,out_dim,kernel_size)
        
    def forward(self, x):
        x = self.up(x)
        x=self.conv(x)
        return x



class Ann_PredictionHead(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.conv=nn.Conv2d(in_dim,1,kernel_size=1)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        x=self.conv(x)
        x=self.sig(x)
        return x



class Ann_Mask_PredictionHead(nn.Module):
    def __init__(self, in_dim=3):
        super().__init__()
        self.conv=nn.Conv2d(in_dim,1,kernel_size=1)
        self.sig=nn.Sigmoid()
    def forward(self, x):
        x=self.conv(x)
        x=self.sig(x)
        return x

