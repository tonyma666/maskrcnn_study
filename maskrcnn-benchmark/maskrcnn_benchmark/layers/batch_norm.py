# Time: 2019-04-03     23:25

import torch 
from torch import nn 

class FrozenBatchNorm2d(nn.Module):
    """
    固定scale和bias的bn层
    为什么固定？因为在batchsize per GPU很小的情况下，如果不采用多GPU同步，BN达不到应有的效果，
    所以干脆使用固定的原pretrained模型的值
    """
    def __init__(self, n):
        # n是通道数目，bn层只于通道数n有关，因为算的是每一个通道上的均值和方差
        super(FrozenBatchNorm2d, self).__init__()
        # register_buffer： 保存module不被视为模型训练参数的参数，并且以后可以把'weight'当做成员变量来使用！
        self.register_buffer('weight', torch.ones(n)) 
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def forward(self, x):
        self.weight = self.weight.view(1, -1, 1, 1)
        self.bias = self.bias.view(1, -1, 1, 1)
        self.running_mean = self.running_mean.view(1, -1, 1, 1)
        self.running_var = self.running_var.view(1, -1, 1, 1)
        return self.weight * ((x - self.running_mean) * self.running_var.rsqrt()) + self.bias  # broadcasting + 归一化 + 仿射变换