#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:39:39 2019

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

__all__ = ['ASPNet', 'ASPNet50', 'ASPNet50_v1', 'ASPNet_v2']

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
    
class _ASPPModule_v2b(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256*5 = 1280
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=256, os=8, norm_layer=None):
        super(_ASPPModule_v2b, self).__init__()
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
                
        
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x), self.aspp_4(x)], dim=1)

        
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


class AttentionModule_v1(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes=1024, planes=512, reduction=8, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v1, self).__init__()
        
        # Attention layer
        # GlobalAvgPool
        self.downchannels = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
        
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
                
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
    def forward(self, x):
        # print(x_low.size())
        x = self.downchannels(x)
        channel_attention = self.ca_subAvgpool(x)
        spatial_attention = self.psi(x)
        
        x_out = x + x * channel_attention + x * spatial_attention

        return x_out
    
class AttentionModule_v2(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes=1024, planes=512, reduction=8, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v2, self).__init__()
        
        # Attention layer
        # GlobalAvgPool
        self.downchannels = nn.Sequential(
                nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
        
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
        """
                
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
        """
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
        
    def forward(self, x):
        # print(x_low.size())
        x = self.downchannels(x)
        channel_attention = self.ca_subAvgpool(x)
        spatial_attention = self.psi(x)
        
        x_out = x + x * channel_attention + x * spatial_attention
        
        return x_out
    

    
class AttentionModule_v3(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes=2048, planes=2048, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v3, self).__init__()
        """
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
        """       
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                ) 
                
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
    def forward(self, x):
        channel_attention = self.ca_subAvgpool(x)
        spatial_attention = self.psi(x)
        
        x_out = x + x * channel_attention + x * spatial_attention
        
        return x_out
    
class AttentionModule_v4(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes=2048, planes=2048, reduction=8, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v4, self).__init__()
        
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, groups=planes, bias=True),
                # norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
    def forward(self, x):
        channel_attention = self.ca_subAvgpool(x)
        spatial_attention = self.psi(x)
        
        x_out = x + x * channel_attention + x * spatial_attention
        
        return x_out
    
class AttentionModule_v5(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes_low=1024, inplanes_high=512, planes=512, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v5, self).__init__()
        
        # Attention layer
        # GlobalAvgPool
        self.downchannels = nn.Sequential(
                nn.Conv2d(in_channels=inplanes_low, out_channels=planes, 
                          kernel_size=3, stride=1, padding=4, dilation=4, 
                          bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
                
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes * 2, out_channels=int(planes // reduction), 
                          kernel_size=3, stride=1, padding=0, dilation=1, 
                          groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )

        self.psi = nn.Sequential(
                nn.Conv2d(inplanes_high, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
        self.fusion = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dialtion=1, groups=1, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x_low, x_high):
        # print(x_low.size())
        x_low = self.downchannels(x_low)
        assert x_low.size()[2:3] == x_high.size()[2:3], "{0} vs {1}".format(x_low.size(), x_high.size())
        x = torch.cat([x_low, x_high], dim=1)  # Channels: 512 + 512 =1024
        channel_attention = self.ca_subAvgpool(x)
        
        spatial_attention = self.psi(x_high)
        
        x_f = x_low * channel_attention + x_low * spatial_attention
        x_f = self.fusion(x_f)
       
        x_out = x_low + x_f

        return x_out
    
class AttentionModule_v6(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes_low=2048, inplanes_high=512, planes=512, reduction=1, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v6, self).__init__()
        
        # Attention layer
        # GlobalAvgPool
        self.featurefusion = nn.Sequential(
                nn.Conv2d(in_channels= inplanes_low + inplanes_high, out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, 
                          bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
                  
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=3, stride=1, padding=0, dilation=1, 
                          groups=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
                
        """
                
        self.ca_subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=planes, out_channels=planes, 
                          kernel_size=3, stride=1, padding=0, dilation=1, 
                          groups=planes, bias=True),
                norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=planes, out_channels=int(planes // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // reduction), out_channels=planes, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
        """
        

        self.psi = nn.Sequential(
                nn.Conv2d(inplanes_high, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.Sigmoid()
                )
        
        self.fusion = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, dialtion=1, groups=1, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True)
                )
        
    def forward(self, x_low, x_high):
        # print(x_low.size())
        x_low = self.downchannels(x_low)
        assert x_low.size()[2:3] == x_high.size()[2:3], "{0} vs {1}".format(x_low.size(), x_high.size())
        x = torch.cat([x_low, x_high], dim=1)  # Channels: 512 + 512 =1024
        x = self.featurefusion(x)
        channel_attention = self.ca_subAvgpool(x)
        
        spatial_attention = self.psi(x_high)
        
        x_f = x_low * channel_attention + x_low * spatial_attention
        x_f = self.fusion(x_f)
       
        x_out = x_low + x_f

        return x_out
    
class AttentionModule_v7(nn.Module):
    """
    Channel Attention Module
    """
    def __init__(self, inplanes_low=2048, inplanes_high=512, outplanes=512, reduction=8, norm_layer=None):  # reduction=1, 4, 8, 16
        super(AttentionModule_v7, self).__init__()
        # Attention layer
        # GlobalAvgPool
        self.downchannel = nn.Sequential(
                nn.Conv2d(inplanes_low, inplanes_high, kernel_size=3, stride=1,
                          padding=1, dilation=1, bias=False),
                norm_layer(inplanes_high),
                nn.ReLU(inplace=True)
                )
        
        self.subAvgpool3x3 = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=inplanes_high, out_channels=inplanes_high, 
                          kernel_size=3, stride=1, padding=0, dilation=1, 
                          groups=inplanes_high, bias=True),
                # norm_layer(inplanes_high),
                nn.ReLU(inplace=True)
                )
        
        self.subAvgpool5x5 = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(5,5)),  # GlobalAvgPool 3, 5, 7
                nn.Conv2d(in_channels=inplanes_high, out_channels=inplanes_high, 
                          kernel_size=5, stride=1, padding=0, dilation=1,
                          groups=inplanes_high, bias=True),
                # norm_layer(inplanes_high),
                nn.ReLU(inplace=True)
                )
                
        self.ca = nn.Sequential(
                nn.Conv2d(in_channels=inplanes_high, out_channels=int(inplanes_high // reduction), 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(inplanes_high // reduction), out_channels=inplanes_high, 
                          kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                nn.Sigmoid() # nn.Softmax(dim=1) or nn.Sigmoid()
                )
                
        self.psi = nn.Sequential(
                nn.Conv2d(inplanes_high, 1, kernel_size=1, stride=1, 
                          padding=0, dilation=1, bias=False),  # padding=0, dilation=1
                nn.ReLU(inplace=True)
                )
        
        self.featurefusion = nn.Sequential(
                nn.Conv2d(inplanes_high * 3, outplanes, kernel_size=1, stride=1, 
                          padding=0, dilation=1, groups=1, bias=False),
                norm_layer(outplanes),
                nn.ReLU(inplace=True)
                )
        
        
    def forward(self, x):
        x = self.downchannel(x)
        
        x_3 = self.subAvgpool3x3(x)
        x_5 = self.subAvgpool5x5(x)
        x_add = x_3 + x_5
        channel_attention = self.ca(x_add)
        
        spatial_attention = self.psi(x)
        
        x_ca = x * channel_attention
        x_sa = x * spatial_attention
        
       
        x_out = x+ x_ca + x_sa
        
        return x_out

#############################################################                        
class ASPNet(nn.Module):
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
        super(ASPNet, self).__init__()
        
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
        
        # self.attention = self._make_attention_layer(AttentionModule_v1, 1024, 512, 8, norm_layer=norm_layer )
        self.attention = self._make_attention_layer(AttentionModule_v2, 1024, 512, 8, norm_layer=norm_layer )
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True), # 512+64+64
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                # nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),   # change
                # nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
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
    
    def _make_attention_layer(self, block, inplanes, planes, reduction, norm_layer):
        return block(inplanes, planes, reduction, norm_layer)
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x_attention = self.attention(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        assert x_attention.size()[2:3] == x.size()[2:3], "{0} vs {1}".format(x_attention.size(), x.size())
        x_last = torch.cat([x, x_attention], dim=1)
        
        x_last = self.conv_last(x_last)
        
        """
        x_ori_size = x_ori.size()
        x_size = x.size()
        if not (x_size[2] == x_ori_size[2] and x_size[3] == x_ori_size[3]):
            x = F.interpolate(x, size=(x_ori_size[2], x_ori_size[3]), mode='bilinear',  align_corners=True) # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x.size(), x_ori.size())
        
        # x = self.conv_last(x)
        """
        
        return x_last
    
class ASPNet_v1(nn.Module):
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
        super(ASPNet_v1, self).__init__()
        
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
        
        
        # self.attention = self._make_attention_layer(AttentionModule_v3, 1024, 1024, 1, norm_layer=norm_layer )
        self.attention = self._make_attention_layer(AttentionModule_v3, 2048, 2048, 1, norm_layer=norm_layer )
        # self.attention = self._make_attention_layer(AttentionModule_v4, 2048, 2048, 8, norm_layer=norm_layer )
        
        self.conv_last = nn.Sequential(
                nn.Dropout2d(0.2),
                # nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),   # change
                # nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
                )
        
        """
        self.conv_last = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True), # 512+64+64
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                # nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),   # change
                # nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
                )
        """
        
        
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
    
    def _make_attention_layer(self, block, inplanes, planes, reduction, norm_layer):
        return block(inplanes, planes, reduction, norm_layer)
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # x = self.attention(x)
        x = self.layer4(x)
        x = self.attention(x)
        x = self.layer5(x)
        # x = self.attention(x)
        
        x_last = self.conv_last(x)
        
        """
        x_ori_size = x_ori.size()
        x_size = x.size()
        if not (x_size[2] == x_ori_size[2] and x_size[3] == x_ori_size[3]):
            x = F.interpolate(x, size=(x_ori_size[2], x_ori_size[3]), mode='bilinear',  align_corners=True) # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x.size(), x_ori.size())
        
        # x = self.conv_last(x)
        """
        
        return x_last
    
class ASPNet_v2(nn.Module):
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
        super(ASPNet_v2, self).__init__()
        
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
        
        self.attention = self._make_attention_layer(AttentionModule_v5, 2048, 512, 512, 1, norm_layer=norm_layer )
        # self.attention = self._make_attention_layer(AttentionModule_v2, 1024, 512, 8, norm_layer=norm_layer )
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True), # 512+64+64
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                # nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),   # change
                # nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
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
    
    def _make_attention_layer(self, block, inplanes_low, inplanes_high, planes, reduction, norm_layer):
        return block(inplanes_low, inplanes_high, planes, reduction, norm_layer)
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x_low = x
        x = self.layer4(x)
        x_low = x
        x = self.layer5(x)
        
        x_attention = self.attention(x_low, x)
        
        assert x_attention.size()[2:3] == x.size()[2:3], "{0} vs {1}".format(x_attention.size(), x.size())
        x_last = torch.cat([x, x_attention], dim=1)
        x_last = self.conv_last(x_last)
        
        return x_last
    
class ASPNet_v3(nn.Module):
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
        super(ASPNet_v3, self).__init__()
        
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
        
        self.attention = self._make_attention_layer(AttentionModule_v1, 1024, 512, 8, norm_layer=norm_layer )
        # self.attention = self._make_attention_layer(AttentionModule_v2, 1024, 512, 8, norm_layer=norm_layer )
        
        self.skip = nn.Sequential(
                nn.Conv2d(512, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(48),
                nn.ReLU(inplace=True)
                )
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(1072, 512, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True), # 512+64+64
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                # nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),   # change
                # nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
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
    
    def _make_attention_layer(self, block, inplanes, planes, reduction, norm_layer):
        return block(inplanes, planes, reduction, norm_layer)
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        
        x = self.layer2(x)
        x_skip = self.skip(x)
        
        x = self.layer3(x)
        x_attention = self.attention(x)
        
        x = self.layer4(x)
        x = self.layer5(x)
        
        assert x_attention.size()[2:3] == x.size()[2:3], "{0} vs {1}".format(x_attention.size(), x.size())
        x_cat = torch.cat([x, x_attention], dim=1)
        
        x_cat_size = x_cat.size()   # [b, 1024, w/16, h/16]
        x_skip_size = x_skip.size() # [b, 48, w/8, h/8]
        if not (x_cat_size[2] == x_skip_size[2] and x_cat_size[3] == x_skip_size[3]):
            x_cat = F.interpolate(x_cat, size=(x_skip_size[2], x_skip_size[3]), mode='bilinear', align_corners=True) # umsample
            # x5 = nn.functional.upsample(x5, size=(x2_size[2], x2_size[3]), mode='bilinear', align_corners=True)
        assert x_cat.size()[2:3] == x_skip.size()[2:3], "{0} vs {1}".format(x_cat.size(), x_skip.size())
        
        x_last = torch.cat([x_cat, x_skip], dim=1)  # [b, 1024+48, w/8, h/8]
        x_last = self.conv_last(x_last)
        
        return x_last
        
##############################################################################   
def ASPNet50(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet101(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet152(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet50_v1(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet_v1(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet101_v1(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet_v1(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet152_v1(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet_v1(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet50_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet101_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def ASPNet152_v2(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = ASPNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model

        
if __name__ == '__main__':
    model = ASPNet50(num_classes=150)