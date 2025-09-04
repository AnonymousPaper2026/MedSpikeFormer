import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


class Ann_ConvBn(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=3//2,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.sig=nn.Sigmoid()
        
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.sig(x)
        return x
    


class Ann_DSConv(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
        self.pconv=nn.Conv2d(in_dim,out_dim,kernel_size=1)
        
        
    def forward(self, x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        x=self.pconv(x)
        return x
    

class Ann_MLP(nn.Module):
    def __init__(self, in_dim,out_dim,kernel_size):
        super().__init__()
        self.pconv1=nn.Conv2d(in_dim,out_dim,kernel_size=1)
        self.bn1=nn.BatchNorm2d(out_dim)
        self.conv=nn.Conv2d(out_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2,groups=out_dim)
        self.bn2=nn.BatchNorm2d(out_dim)
        self.act1 = nn.GELU()
        self.pconv2=nn.Conv2d(out_dim,out_dim,kernel_size=1)
        
        
    def forward(self, x):
        x=self.pconv1(x)
        x=self.bn1(x)
        x=self.conv(x)
        x=self.bn2(x)
        x=self.act1(x)
        x=self.pconv2(x)
        return x
    
    
class Ann_Norm(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.bn=nn.BatchNorm2d(in_dim)
        
        
    def forward(self, x):
        x=self.bn(x)
        return x


class Ann_Conv(nn.Module):
    def __init__(self,in_dim,kernel_size):
        super().__init__()
        self.conv1=Ann_DSConv(in_dim,in_dim,kernel_size)
        self.conv2=Ann_DSConv(in_dim,in_dim,kernel_size)
        self.bn1=Ann_Norm(in_dim)
        self.bn2=Ann_Norm(in_dim)
        
    def forward(self, x):
        x = self.conv1(x)+x
        x=self.bn1(x)
        x = self.conv2(x)+x
        x=self.bn2(x)
        return x
    
