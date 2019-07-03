#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 12:53:41 2019

@author: hi
"""

import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from lib.nn.modules.batchnorm import SynchronizedBatchNorm2d

# 提供了暴露接口用的“白名单”
__all__ = ['AMDNeXt', 'AMDNext50', 'AMDNext101', 'AMDNext152'] 

affine_par = True

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
    
model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
    }


def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, dilation=dilation, bias=False)


class GroupBottleneck(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(GroupBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
       
        return z
        
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
    
class AttentionGroupBottleneck(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(AttentionGroupBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        #self.dropout = nn.Dropout2d(0.5)
        #self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
        # Attention layer
        
        # GlobalAvgPool
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        # self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        # self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        # self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(
                nn.Linear(in_features=planes, out_features=int(planes // 16), bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=int(planes // 16), out_features=planes, bias=True),
                nn.Sigmoid()
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(1, affine=affine_par),
                nn.Sigmoid()
                )
        
        
    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
       
        return z
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Attention
        original_out = out
        # print(out.size())
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)   # --> 1D   (b, c)
        # print(out.size())
        # out = self.fc1(out)
        # out = self.relu(out)
        # out = self.fc2(out)
        # out = self.sigmoid(out)
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1)  # (b, c, 1, 1)
        # print(out.size())
        # out = self.dropout(out)
       
        # out = out * original_out
        out = original_out * out     # **
        
        out = self.relu(out)
        
        psi = self.psi(out)
        
        # dropout 0.5
        # out = self.dropout(out)
             
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # out += residual
        out += (residual + residual * psi)
        out = self.relu(out)
        
        return out
    
class AttentionGroupBottleneck_v2(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(AttentionGroupBottleneck_v2, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # self.dropout = nn.Dropout2d(0.5)
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
        # Attention layer
        self.att_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool
                nn.Conv2d(in_channels=planes, out_channels=int(planes // 16), kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(int(planes // 16), affine=affine_par),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=int(planes // 16), out_channels=planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes, affine=affine_par),
                nn.Sigmoid()
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(1, affine=affine_par),
                nn.Sigmoid()
                )
        
        
    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
       
        return z
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Attention
        original_out = out
        # print(out.size()) # [3, 512, 50, 50]
        
        out = self.att_pool(out)
        # print(out.size())  # [b, 512, 1, 1]
        # out = self.dropout(out)
       
        # out = out * original_out
        out = original_out * out     # **
        # print(out.size())  # [b, 512, 50, 50]
        out = self.relu(out)
        
        psi = self.psi(out)
        # print(psi.size()) # [b, 1, 50, 50]
             
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # out += residual
        out += (residual + residual * psi)
        # print(out.size()) # [b, 1024, 50, 50]
        out = self.relu(out)
        
        return out   
    
class AttentionGroupBottleneck_v3(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(AttentionGroupBottleneck_v3, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # self.dropout = nn.Dropout2d(0.5)
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
        # Attention layer
        self.att_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                norm_layer(planes, affine=affine_par),
                nn.ReLU(inplace=True)
                )
        
        self.fc = nn.Sequential(
                nn.Linear(in_features=planes, out_features=int(planes // 16), bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=int(planes // 16), out_features=planes, bias=True),
                nn.Sigmoid()
                # nn.Softmax(dim=1) #  (0, 1)  Sum=1
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(1, affine=affine_par)
                # nn.Sigmoid()
                )
        
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=3)
        
        
    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
       
        return z
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Attention
        original_out = out
        # print(out.size()) # [3, 512, 50, 50]
        
        #psi = self.psi(self.relu(out))
        
        # the dim range of [b, c, w, h] : [-4, 3]
        psi_o = self.psi(self.relu(out))
        #print(psi_o.size())  # [b, c, w, h] [b, c, 46, 46]
        psi_2 = self.softmax1(psi_o)
        #print(psi_2)
        psi_3 = self.softmax2(psi_o)
        #print(psi_2)
        psi = psi_2 + psi_3
        #print(psi)
        # psi = self.psi(psi)
        
        #print(psi.size())
        
        out = self.att_pool(out)
        # print(out.size())  # [b, 512//16, 1, 1]
        out = out.view(out.size(0), -1)   # --> 1D   (b, c//16)
        # print(out.size())
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1)  # (b, c, 1, 1)
        # print(out.size())
        # out = self.dropout(out)
        out = original_out * out     # ** # out = out * original_out
        # print(out.size())  # [b, 512, 50, 50]
        out = self.relu(out)
        
        #psi = self.psi(out)
        # print(psi.size()) # [b, 1, 50, 50]
             
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # out += residual
        out += (residual + residual * psi) # change
        # print(out.size()) # [b, 1024, 50, 50]
        out = self.relu(out)
        
        return out   
    
class AttentionGroupBottleneck_v4(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(AttentionGroupBottleneck_v4, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2, affine=affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # self.dropout = nn.Dropout2d(0.5)
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
        # Attention layer
        self.att_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=False),
                norm_layer(planes),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes),
                # nn.Sigmoid()
                nn.Softmax(dim=1)
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(planes, 1, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(1)
                # nn.Sigmoid()
                )
        
        self.softmax1 = nn.Softmax(dim=2)
        self.softmax2 = nn.Softmax(dim=3)
        
        
    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i] + y[i])
       
        return z
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Attention
        original_out = out
        # print(out.size()) # [3, 512, 50, 50]
        
        #psi = self.psi(self.relu(out))
        # the dim range of [b, c, w, h] : [-4, 3]
        psi_o = self.psi(self.relu(out))
        #print(psi_o.size())  # [b, c, w, h] [b, c, 24, 34]
        psi_w = self.softmax1(psi_o)
        #print(psi_2)
        psi_h = self.softmax2(psi_o)
        #print(psi_2)
        psi = psi_w + psi_h
        #print(psi)
        #print(psi.size())  # [b, 1, 50, 50]
        
        out = self.att_pool(out)
        
        out = original_out * out     # ** # out = out * original_out
        out = self.relu(out)
             
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # out += residual
        out += (residual + residual * psi) # change
        # print(out.size()) # [b, 1024, 50, 50]
        out = self.relu(out)
        
        return out   
    
    
class _ASPPModule(nn.Module):
    """"
    : param in_planes: default = 2048
    : param out_planes: default = 256
    : param os: 8 0r 16
    """
    def __init__(self, in_planes=2048, out_planes=256, os=8, norm_layer=None):
        super(_ASPPModule, self).__init__()
        if os == 8:
            rates = [1, 6, 12, 18]
        elif os == 16:
            rates = [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                # nn.AdaptiveAvgPool2d(rates[0]),
                nn.AdaptiveAvgPool2d((3, 3)),
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
        
        self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x)], dim=1)
        x = self.aspp_catdown(x)
        x = self.dropout(x)
        
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
            rates = [3, 6, 12, 18]   # [1, 6, 12, 18]
        elif os == 16:
            rates = [3, 12, 24, 36]  # [1, 12, 24, 36]
        else:
            raise NotImplementedError
        
        self.gave_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(rates[0]),  # nn.AdaptiveAvgPool2d((3, 3)),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=True),
                norm_layer(out_planes),
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
                norm_layer(2 * out_planes),  # 512 256
                nn.ReLU(inplace=True)
                )
        
        # self.dropout = nn.Dropout2d(np.random.choice([0., 0.1, 0.2, 0.3, 0.4, 0.5], 1).item())
        
    def forward(self, x):
        size = x.size()
        x1 = F.interpolate(self.gave_pool(x), (size[2], size[3]), mode='bilinear', align_corners=True) # umsample
        x = torch.cat([x1, self.conv1x1(x), self.aspp_1(x), self.aspp_2(x), self.aspp_3(x)], dim=1)
        x = self.aspp_catdown(x)
        # x = self.dropout(x)
        
        return x

class _PyramidPoolingModule(nn.Module):
    """
    Note: 
        norm_layer :SynchronizedBatchNorm2d  | nn.BatchNorm2d
    
    reduction_planes=512
    """
    def __init__(self, inplanes, pool_series, norm_layer=None):
        super(_PyramidPoolingModule, self).__init__()
        self.psp = []
        
        for scale in pool_series:
            self.psp.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(inplanes, 512, kernel_size=1, bias=False),
                    # nn.BatchNorm2d(reduction_planes, momentum=0.95)
                    # SynchronizedBatchNorm2d(reduction_planes)
                    norm_layer(512, momentum=0.95),
                    # norm_layer(512),
                    nn.ReLU(inplace=True)
                    ))
            
        self.psp = nn.ModuleList(self.psp)
        
    def forward(self, x):
        input_size = x.size()
        # print(input_size)
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(F.upsample(
                    pool_scale(x),
                    (input_size[2], input_size[3]),
                    mode='bilinear', align_corners=False))
        
        out = torch.cat(psp_out, 1)
        
        return out
    
#######################################################################    
class AMDNeXt(nn.Module):
    """
    Dilated Pre-trained ResNet Model, which preduces the stride of 8 featuremaps at conv5
    
    Parameters
    -------------------------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 150
        Number of classification classes.
    dilated : bool, default False
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
    
    def __init__(self, block, attentionblock, layers, num_classes, norm_layer=None, groups=32, dilate_scale=16):
        self.inplanes = 128
        super(AMDNeXt, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 128)
        self.bn3 = norm_layer(128)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 128, layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2, groups=groups, norm_layer=norm_layer)
        
        if dilate_scale == 8:
            self.layer3 = self._make_layer(attentionblock, 512, layers[2], stride=1,
                                           dilation=2, groups=groups, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 1024, layers[3], stride=1,
                                           dilation=4, groups=groups, norm_layer=norm_layer)
        elif dilate_scale == 16:
            self.layer3 = self._make_layer(attentionblock, 512, layers[2], stride=2,
                                           groups=groups, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 1024, layers[3], stride=1,
                                           dilation=2, groups=groups, norm_layer=norm_layer)   
        else:
            self.layer3 = self._make_layer(attentionblock, 512, layers[2], stride=2,
                                           groups=groups, norm_layer=norm_layer)
            self.layer4 = self._make_layer(block, 1024, layers[3], stride=2,
                                           groups=groups, norm_layer=norm_layer)
            
        # self.layer5 = self._make_aspp_layer(_ASPPModule, 2048, 256, os=8, norm_layer=norm_layer)
        self.layer5 = self._make_aspp_layer(_ASPPModule_v2, 2048, 256, os=8, norm_layer=norm_layer)
        
        self.one_skip1 = nn.Sequential(
                nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True)
                )
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(576, 512, kernel_size=3, padding=1, dilation=1, bias=False), # 512+64+64
                norm_layer(512),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.2),
                nn.Dropout2d(np.random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5], 1).item()),
                nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                )
        
        self.conv_branch = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(512, num_classes, kernel_size=1, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                #nn.init.kaiming_normal_(m.weight.data)
                
                m.weight.data.normal_(0, 0.001)
                """
                if m.bias is not None:
                    m.bias.data.zero_()
                """
                    
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.001)
                #nn.init.kaiming_normal_(m.weight.data)
                #if m.bias is not None:
                #    m.bias.data.zero_()
                # m.weight.data.normal_(0, 0.001)
                
            elif isinstance(m, norm_layer):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0) # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, groups=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 
                              kernel_size=1, stride=stride, bias=False),
                    norm_layer(planes * block.expansion),
                    )
                    
        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.inplanes, planes, stride, dilation=1,
                                groups=groups, downsample=downsample, previous_dilation=dilation, 
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2, 
                                groups=groups, downsample=downsample, previous_dilation=dilation, 
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unkown dilation size: {}".format(dilation))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, groups=groups, 
                                previous_dilation=dilation, norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def _make_psp_layer(self, block, inplanes, pool_series, norm_layer):
        return block(inplanes, pool_series, norm_layer)
    
    def _make_aspp_layer(self, block, inplanes, planes, os, norm_layer):
        return block(inplanes, planes, os, norm_layer)
    
    
    def forward(self, x):  # (b, 3, w, h)
        x_ori = x
        x = self.relu1(self.bn1(self.conv1(x))) # -> (b, 64, w/2, h/2)
        x = self.relu2(self.bn2(self.conv2(x))) # -> (b, 64, w/2, h/2)
        x = self.relu3(self.bn3(self.conv3(x))) # -> (b, 64, w/2, h/2)
        x = self.maxpool(x)  # -> (b, 64, w/4, h/4)
        
        # x0 = x   # -> (b, 64, w/4, h/4)
        x1 = self.layer1(x)    # -> (b, 256, w/4, h/4)
        x2 = self.layer2(x1)    # -> (b, 512, w/8, h/8)
        x3 = self.layer3(x2)    # -> (b, 1024, w/8, h/8)  or -> (b, 1024, w/16, h/16)
        x4 = self.layer4(x3)    # -> (b, 2048, w/8, h/8)  or -> (b, 1024, w/16, h/16)
        x5 = self.layer5(x4)    # -> (b, 256, w/8, h/8)  or -> (b, 256, w/16, h/16)
        
        # x_branch = self.conv_branch(x2)
        
        x_s1 = self.one_skip1(x1)
        
      
        # Upsample x2 / x4
        x_size = x_ori.size()
        x_s1_size = x_s1.size()
        
        x5_size = x5.size()
        if not (x5_size[2] == x_s1_size[2] and x5_size[3] == x_s1_size[3]):
            x5 = F.interpolate(x5, size=(x_s1_size[2], x_s1_size[3]), mode='bilinear') # umsample
            # x2 = nn.functional.upsample(x2, size=(x0_size[2], x0_size[3]), mode='bilinear')
        assert x5.size()[2:3] == x_s1.size()[2:3], "{0} vs {1}".format(x5.size(), x_s1.size())
        
        x_last = torch.cat([x5, x_s1], dim=1)
        x_last = self.conv_last(x_last)   # !!!
        
        #x_branch = self.conv_branch(x3)  # !!!
        
        x_last_size = x_last.size()
        if not (x_last_size[2] == x_size[2] and x_last_size[3] == x_size[3]):
            x_last = F.interpolate(x_last, size=(x_size[2], x_size[3]), mode='bilinear') # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x_last.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x_last.size(), x_ori.size())
        
        # branch upsample 16
        x_branch = self.conv_branch(x3)  # !!!
        x_branch_size = x_branch.size()
        if not (x_branch_size[2] == x_size[2] and x_branch_size[3] == x_size[3]):
            x_branch = F.interpolate(x_branch, size=(x_size[2], x_size[3]), mode='bilinear') # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x_branch.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x_branch.size(), x_ori.size())
        
        return x_last, x_branch
    

###################################################################
# Models   


def AMDNext50(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AMDNeXt(GroupBottleneck, AttentionGroupBottleneck_v4, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def AMDNext101(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AMDNeXt(GroupBottleneck, AttentionGroupBottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def AMDNext152(pretrained=False,  num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AMDNeXt(GroupBottleneck, AttentionGroupBottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


def load_url(url, model_dir='./pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return torch.load(cached_file, map_location=map_location)

if __name__ == '__main__':
    model = AMDNext50(pretrained=False)
    torch.save(model, './pretrained/atdnext50.pth')