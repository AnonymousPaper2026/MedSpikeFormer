import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode


class Quant_ConvBn(nn.Module):
    def __init__(self, in_dim,out_dim,T,kernel_size):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=kernel_size//2,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.pconv=nn.Conv2d(in_dim,out_dim,kernel_size=1)
        self.bn=nn.BatchNorm2d(out_dim)
        self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.T=T
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.conv(x)
        x=self.bn(x)
        x=x.view(self.T,B,C,H,W)
        x=self.act(x)
        return x



class Quant_DSConv(nn.Module):
    def __init__(self, in_dim,out_dim,T,kernel_size):
        super().__init__()
        groups = out_dim if in_dim > out_dim else in_dim
        self.conv=nn.Conv2d(in_dim,out_dim,kernel_size=kernel_size,padding=1,groups=groups)
        self.bn=nn.BatchNorm2d(out_dim)
        self.act = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.pconv=nn.Conv2d(in_dim,out_dim,kernel_size=1)
        # self.act2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.T=T
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.conv(x)
        x=self.bn(x)
        x=x.view(self.T,B,C,H,W)
        x=self.act(x)
        x=x.view(self.T*B,C,H,W)
        x=self.pconv(x)
        x=x.view(self.T,B,C,H,W)
        # x=self.act2(x)
        return x
    

class Quant_MLP(nn.Module):
    def __init__(self, in_dim,out_dim,T):
        super().__init__()
        self.pconv1=nn.Conv2d(in_dim,out_dim,kernel_size=1)
        self.bn1=nn.BatchNorm2d(out_dim)
        self.conv=nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1,groups=out_dim)
        self.bn2=nn.BatchNorm2d(out_dim)
        self.act1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.pconv2=nn.Conv2d(out_dim,out_dim,kernel_size=1)
        # self.act2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="torch")
        self.T=T
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.pconv1(x)
        x=self.bn1(x)
        x=self.conv(x)
        x=self.bn2(x)
        Tb,C,H,W=x.shape
        B=Tb//self.T
        x=x.view(self.T,B,C,H,W)
        x=self.act1(x)
        x=x.view(self.T*B,C,H,W)
        x=self.pconv2(x)
        x=x.view(self.T,B,C,H,W)
        # x=self.act2(x)
        return x
    
    
class Quant_Norm(nn.Module):
    def __init__(self, in_dim,T):
        super().__init__()
        self.T=T
        self.bn=nn.BatchNorm2d(in_dim)
        
        
    def forward(self, x):
        _,B,C,H,W=x.shape
        x=x.view(self.T*B,C,H,W)
        x=self.bn(x)
        x=x.view(self.T,B,C,H,W)
        return x
    
    
class Quant_Spike_Tensor(nn.Module):
    def __init__(self,T):
        super().__init__()
        self.T=T
    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        return x
    




class Quant_Spike_Conv(nn.Module):
    def __init__(self,in_dim,out_dim,T,kernel_size):
        super().__init__()
        self.conv1=Quant_DSConv(in_dim,out_dim,T,kernel_size)
        self.conv2=Quant_DSConv(in_dim,out_dim,T,kernel_size)
        self.bn1=Quant_Norm(out_dim,T)
        self.bn2=Quant_Norm(out_dim,T)
        
    def forward(self, x):
        x = self.conv1(x)+x
        x=self.bn1(x)
        x = self.conv2(x)+x
        x=self.bn2(x)
        return x
    
