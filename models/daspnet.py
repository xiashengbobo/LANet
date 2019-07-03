#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 11:47:55 2019

@author: hi
"""


#import os
#import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.utils.model_zoo as model_zoo
#from torchvision import models

from lib.nn.modules.batchnorm import SynchronizedBatchNorm2d 

__all__ = ['DASPNet', 'DASPNet50', 'DASPNet101', 'DASPNet152']

affine_par = True

##############################################################
def outS(i):
    i = int(i)
    i = (i+1) / i
    i = int(np.ceil((i+1) / 2.0) )
    i = (i+1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock
    affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数
    inplace: 选择是否进行覆盖运算
    """
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               paddig=1, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        out = self.relu(out)
        
        return out
    
class Bottleneck(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change !!! stride
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=padding, dilation = dilation, bias=False)   # chang !!!
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
   
        
class _ASPPModule(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=512, os=8, norm_layer=None):
        super(_ASPPModule, self).__init__()
        if os == 8:
            rates = [1, 6, 12, 18]
        elif os == 16:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True),
                norm_layer(out_planes, momentum=0.95),
                nn.ReLU(inplace=True)
                )
        
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
                )
        
        self.aspp_1 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                          padding=rates[1], dilation=rates[1], bias=True),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_2 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                          padding=rates[2], dilation=rates[2], bias=True),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_3 = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                          padding=rates[3], dilation=rates[3], bias=True),
                norm_layer(out_planes),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_catdown = nn.Sequential(
                nn.Conv2d(5 * out_planes, 2 * out_planes, kernel_size=1, bias=False),
                norm_layer(2 * out_planes),  # 512
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x)], dim=1)
        x = self.aspp_catdown(x)
        
        return x
    
class _ASPPModule_v2(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=256, os=8, norm_layer=None):
        super(_ASPPModule_v2, self).__init__()
        if os == 8:
            rates = [3, 2, 6, 12, 18]   # [1, 1, 6, 12, 18]
        elif os == 16:
            rates = [3, 1, 12, 24, 36]  # [1, 1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),  # nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_planes, 128, kernel_size=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
        
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_planes, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
                )
        
        self.aspp_1 = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1,
                          padding=rates[1], dilation=rates[1], bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_2 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[2], dilation=rates[2], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_3 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[3], dilation=rates[3], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_4 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[4], dilation=rates[4], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_catdown = nn.Sequential(
                nn.Conv2d(5 * out_planes, 2 * out_planes, kernel_size=1, bias=False),
                norm_layer(2 * out_planes),  #  256 * 2
                nn.ReLU(inplace=True)
                )
        
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x), self.aspp_4(x)], dim=1)
        x = self.aspp_catdown(x)
        # x = self.dropout(x)
        
        return x
    
class _ASPPModule_v3(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=256, os=8, norm_layer=None):
        super(_ASPPModule_v3, self).__init__()
        if os == 8:
            rates = [3, 2, 6, 12, 18]   # [1, 1, 6, 12, 18]
        elif os == 16:
            rates = [3, 2, 12, 24, 36]  # [1, 1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),  # nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_planes, 128, kernel_size=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
        
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_planes, 384, kernel_size=1, bias=False),
                norm_layer(384),
                nn.ReLU(inplace=True)
                )
        
        self.aspp_1 = nn.Sequential(
                nn.Conv2d(in_planes, 320, kernel_size=3, stride=1,
                          padding=rates[1], dilation=rates[1], bias=False),
                norm_layer(320),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_2 = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1,
                          padding=rates[2], dilation=rates[2], bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_3 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[3], dilation=rates[3], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_4 = nn.Sequential(
                nn.Conv2d(in_planes, 64, kernel_size=3, stride=1,
                          padding=rates[4], dilation=rates[4], bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_catdown = nn.Sequential(
                nn.Conv2d(5 * out_planes, 2 * out_planes, kernel_size=1, bias=False),
                norm_layer(2 * out_planes),  #  256 * 2
                nn.ReLU(inplace=True)
                )
        
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x), self.aspp_4(x)], dim=1)
        x = self.aspp_catdown(x)
        # x = self.dropout(x)
        
        return x
    
class _ASPPModule_v4(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=256, os=8, norm_layer=None):
        super(_ASPPModule_v4, self).__init__()
        if os == 8:
            rates = [2, 3, 6, 12, 18]   # [1, 1, 6, 12, 18]
        elif os == 16:
            rates = [2, 3, 12, 24, 36]  # [1, 1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool1 = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),  # nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_planes, 128, kernel_size=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
        
        self.gave_pool2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[1]),  # nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_planes, 128, kernel_size=1, bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
        
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_planes, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_1 = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1,
                          padding=rates[2], dilation=rates[2], bias=False),
                norm_layer(256),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_2 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[3], dilation=rates[3], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_3 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[4], dilation=rates[4], bias=False),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_catdown = nn.Sequential(
                nn.Conv2d(5 * out_planes, 2 * out_planes, kernel_size=1, bias=False),
                norm_layer(2 * out_planes),  #  256 * 2
                nn.ReLU(inplace=True)
                )
        
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool1(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        x2 = F.interpolate(self.gave_pool2(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        
        x = torch.cat([x1, x2, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x)], dim=1)
        x = self.aspp_catdown(x)
        # x = self.dropout(x)
        
        return x
    
class _ASPPModule_v5(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=512, os=8, norm_layer=None):
        super(_ASPPModule_v5, self).__init__()
        if os == 8:
            rates = [1, 6, 12, 18]
        elif os == 16:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),
                nn.Conv2d(in_planes, 256, kernel_size=1, bias=True),
                norm_layer(256, momentum=0.95),
                nn.ReLU(inplace=True)
                )
        
        self.conv1x1 = nn.Sequential(
                nn.Conv2d(in_planes, 512, kernel_size=1, bias=True),
                norm_layer(512),
                nn.ReLU(inplace=True)
                )
        
        self.aspp_1 = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1,
                          padding=rates[1], dilation=rates[1], bias=True),
                norm_layer(256),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_2 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[2], dilation=rates[2], bias=True),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_3 = nn.Sequential(
                nn.Conv2d(in_planes, 128, kernel_size=3, stride=1,
                          padding=rates[3], dilation=rates[3], bias=True),
                norm_layer(128),
                nn.ReLU(inplace=True)
                )
                
        self.aspp_catdown = nn.Sequential(
                nn.Conv2d(5 * out_planes, 2 * out_planes, kernel_size=1, bias=False),
                norm_layer(2 * out_planes),  # 512
                nn.ReLU(inplace=True)
                )
    
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x)], dim=1)
        x = self.aspp_catdown(x)
        
        return x
    
    
class ChannelsAttentionModule_v1(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2948
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v1, self).__init__()
        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1,stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x
    
class ChannelsAttentionModule_v2(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v2, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1,stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x
    
class ChannelsAttentionModule_v3(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v3, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(5,5)),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=5, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1,stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x
    
class ChannelsAttentionModule_v4(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v4, self).__init__()

        self.channelsattention_g = nn.Sequential(
                 nn.AdaptiveAvgPool2d(output_size=(7,7)),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=7, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1,stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x
    
class ChannelFusion(nn.Module):
    """
    Channel Fusion
    """
    def __init__(self, inplanes=256, planes=512, norm_layer=None): 
        super(ChannelFusion, self).__init__()
        self.upchannels = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=inplanes * 2, kernel_size=1,
                          stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)  # nn.ReLU(inplace=True) # nn.Sigmoid()
                )
                
    def forward(self, x_low, x_high):
        x_low = self.upchannels(x_low)
        
        assert x_low.size()[1:3] == x_high.size()[1:3], "{0} vs {1}".format(x_low.size(), x_high.size())
        x_f = x_low + x_high
        
        return x_f
        
 
class ChannelsAttentionModule_v1b(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v1b, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1,1)),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x    
    
class ChannelsAttentionModule_v2b(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v2b, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x   
    
class ChannelsAttentionModule_v2c(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v2b, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x   
class ChannelsAttentionModule_v3b(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v3b, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(5,5)),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=5, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x   
    
class ChannelsAttentionModule_v4b(nn.Module):
    """
    Channel Attention Module : C: 256 512 1024 2048
    """
    def __init__(self, planes=256, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(ChannelsAttentionModule_v4b, self).__init__()

        self.channelsattention_g = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(7,7)),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=7, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x):
        x = self.channelsattention_g(x)
        
        return x   
    
class ChannelFusion_v2(nn.Module):
    """
    Channel Fusion
    """
    def __init__(self, inplanes=256, planes=512, up=2, reduction=1, norm_layer=None): 
        super(ChannelFusion_v2, self).__init__()
        self.upchannels = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=inplanes * up, kernel_size=1,
                          stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True)  # nn.ReLU(inplace=True) # nn.Sigmoid()
                )   
                
        self.channelsfusion= nn.Sequential(
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid()
                )

        
    def forward(self, x_low, x_high):
        x_low = self.upchannels(x_low)
        
        assert x_low.size()[1:3] == x_high.size()[1:3], "{0} vs {1}".format(x_low.size(), x_high.size())
        x_add = x_low + x_high
        
        x_f = self.channelsfusion(x_add)
        
        return x_add, x_f

#############################################################                        
class DASPNet(nn.Module):
    """
    Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5
    
    Parameters
    -------------------------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 150
        Number of classification classes.
    dilated_scale :  default 8
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    ***
    norm_layer : object   [ nn.BatchNorm2d, SynchronizedBatchNorm2d ]
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." 
            Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    
    """
    
    def __init__(self, block, layers, num_classes, norm_layer=None, dilate_scale=16):
        self.inplanes = 64
        super(DASPNet, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = norm_layer(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        
        if dilate_scale == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilate_scale == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)   
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        
        # self.layer5 = self._make_aspp_layer(_ASPPModule, 2048, 256, os=8, norm_layer=norm_layer)
        self.layer5 = self._make_aspp_layer(_ASPPModule_v2, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v3, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v4, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v5, 2048, 256, os=8, norm_layer=norm_layer)
         
        self.ca1 = self._make_channelattention_layer(ChannelsAttentionModule_v2, 256, 1, norm_layer=norm_layer)
        self.ca2 = self._make_channelattention_layer(ChannelsAttentionModule_v2, 512, 1, norm_layer=norm_layer)
        self.ca3 = self._make_channelattention_layer(ChannelsAttentionModule_v2, 1024, 1, norm_layer=norm_layer)
        self.ca4 = self._make_channelattention_layer(ChannelsAttentionModule_v2, 2048, 1, norm_layer=norm_layer)
        
        self.cf12 = self._make_channelfusion_layer(ChannelFusion, 256, 512, norm_layer=norm_layer)
        self.cf23 = self._make_channelfusion_layer(ChannelFusion, 512, 1024, norm_layer=norm_layer)
        self.cf34 = self._make_channelfusion_layer(ChannelFusion, 1024, 2048, norm_layer=norm_layer)
        
        
        self.conv_last = nn.Sequential(
                nn.Dropout2d(0.2),
                nn.Conv2d(512, num_classes, kernel_size=1)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, norm_layer):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                    )
                    
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def _make_aspp_layer(self, block, inplanes, planes, os, norm_layer):
        return block(inplanes, planes, os, norm_layer)
    
    def _make_channelattention_layer(self, block, planes, reduction, norm_layer):
        return block(planes, reduction, norm_layer)
    
    def _make_channelfusion_layer(self, block, inplanes, planes, norm_layer):
        return block(inplanes, planes, norm_layer)
    
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        ca1 = self.ca1(x)
        x = x + x * ca1
        
        x = self.layer2(x)
        ca2 = self.ca2(x)
        cf = self.cf12(ca1, ca2)
        x = x + x * cf
        
        x = self.layer3(x)
        ca3 = self.ca3(x)
        cf = self.cf23(cf, ca3)
        x = x + x * cf
        
        x = self.layer4(x)
        ca4 = self.ca4(x)
        
        # cf = self.cf12(ca1, ca2)
        # cf = self.cf23(cf, ca3)
        cf = self.cf34(cf, ca4)
        
        x = x + x * cf
        
        x = self.layer5(x)
        x = self.conv_last(x)
        
        """
        x_ori_size = x_ori.size()
        x_size = x.size()
        if not (x_size[2] == x_ori_size[2] and x_size[3] == x_ori_size[3]):
            x = F.interpolate(x, size=(x_ori_size[2], x_ori_size[3]), mode='bilinear',  align_corners=True) # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x.size(), x_ori.size())
        
        # x = self.conv_last(x)
        """
        
        return x
    
class DASPNet_v2(nn.Module):
    """
    Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5
    
    Parameters
    -------------------------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    num_classes : int, default 150
        Number of classification classes.
    dilated_scale :  default 8
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    ***
    norm_layer : object   [ nn.BatchNorm2d, SynchronizedBatchNorm2d ]
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition." 
            Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    
    """
    
    def __init__(self, block, layers, num_classes, norm_layer=None, dilate_scale=16):
        self.inplanes = 64
        super(DASPNet_v2, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = norm_layer(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        
        if dilate_scale == 8:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer)
        elif dilate_scale == 16:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=2, norm_layer=norm_layer)   
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           norm_layer=norm_layer)
        
        # self.layer5 = self._make_aspp_layer(_ASPPModule, 2048, 256, os=8, norm_layer=norm_layer)
        self.layer5 = self._make_aspp_layer(_ASPPModule_v2, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v3, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v4, 2048, 256, os=8, norm_layer=norm_layer)
        # self.layer5 = self._make_aspp_layer(_ASPPModule_v5, 2048, 256, os=8, norm_layer=norm_layer)
         
        self.ca0 = self._make_channelattention_layer(ChannelsAttentionModule_v2b, 64, 1, norm_layer=norm_layer)
        self.ca1 = self._make_channelattention_layer(ChannelsAttentionModule_v2b, 256, 1, norm_layer=norm_layer)
        self.ca2 = self._make_channelattention_layer(ChannelsAttentionModule_v2b, 512, 1, norm_layer=norm_layer)
        self.ca3 = self._make_channelattention_layer(ChannelsAttentionModule_v2b, 1024, 1, norm_layer=norm_layer)
        self.ca4 = self._make_channelattention_layer(ChannelsAttentionModule_v2b, 2048, 1, norm_layer=norm_layer)
        
        self.cf01 = self._make_channelfusion_layer(ChannelFusion_v2, 64, 256, 4, 1, norm_layer=norm_layer)
        self.cf12 = self._make_channelfusion_layer(ChannelFusion_v2, 256, 512, 2, 1, norm_layer=norm_layer)
        self.cf23 = self._make_channelfusion_layer(ChannelFusion_v2, 512, 1024, 2, 1, norm_layer=norm_layer)
        self.cf34 = self._make_channelfusion_layer(ChannelFusion_v2, 1024, 2048, 2, 1, norm_layer=norm_layer)
        
        
        self.conv_last = nn.Sequential(
                nn.Dropout2d(0.2),
                nn.Conv2d(512, num_classes, kernel_size=1)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, norm_layer):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                    )
                    
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def _make_aspp_layer(self, block, inplanes, planes, os, norm_layer):
        return block(inplanes, planes, os, norm_layer)
    
    def _make_channelattention_layer(self, block, planes, reduction, norm_layer):
        return block(planes, reduction, norm_layer)
    
    def _make_channelfusion_layer(self, block, inplanes, planes, up, reduction, norm_layer):
        return block(inplanes, planes, up, reduction, norm_layer)
    
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        
        x = self.maxpool(x)
        ca0 = self.ca0(x)
        
        x = self.layer1(x)
        ca1 = self.ca1(x)
        cua, cf01 = self.cf01(ca0, ca1)
        x = x + x * cf01
        
        x = self.layer2(x)
        ca2 = self.ca2(x)
        cua, cf12 = self.cf12(cua, ca2)
        x = x + x * cf12
        
        x = self.layer3(x)
        ca3 = self.ca3(x)
        cua, cf23 = self.cf23(cua, ca3)
        x = x + x * cf23
        
        x = self.layer4(x)
        ca4 = self.ca4(x)
        cua, cf34 = self.cf34(cua, ca4)  
        x = x + x * cf34
        
        x = self.layer5(x)
        x = self.conv_last(x)
        
        """
        x_ori_size = x_ori.size()
        x_size = x.size()
        if not (x_size[2] == x_ori_size[2] and x_size[3] == x_ori_size[3]):
            x = F.interpolate(x, size=(x_ori_size[2], x_ori_size[3]), mode='bilinear',  align_corners=True) # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x.size(), x_ori.size())
        
        # x = self.conv_last(x)
        """
        
        return x
        
############################################################################   
def DASPNet50(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def DASPNet101(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def DASPNet152(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model

def DASPNet50_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet_v2(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def DASPNet101_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet_v2(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def DASPNet152_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = DASPNet_v2(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model
        
if __name__ == '__main__':
    model = DASPNet50(num_classes=150)