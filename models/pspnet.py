#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 17:21:26 2018

@author: sunshine
"""
# PSPNet

# import os
# import sys
# import math
# import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
# from torchvision import models

from lib.nn.modules.batchnorm import SynchronizedBatchNorm2d 

__all__ = ['PSPNet', 'PSPNet50', 'PSPNet101', 'PSPNet152']

affine_par = True

##############################################################
"""
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

    
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    }


def initialize_weight(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(model, nn.BatchNorm2d):
                #  weight , bias -- Variable
                module.weight.data.fill_(1)
                module.bias.data.zero_()
"""
                


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, dilation=1, bias=False)


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock
    """
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
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
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False) # change
        self.bn1 = norm_layer(planes, affine=affine_par)
        
        #padding = dilation  # padding == dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, bias=False)
        self.bn2 = norm_layer(planes, affine=affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 4, affine=affine_par)
        
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
                    #nn.BatchNorm2d(reduction_planes, momentum=0.95)
                    #SynchronizedBatchNorm2d(reduction_planes)
                    norm_layer(512, momentum=0.95),
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

#############################################################                        
class PSPNet(nn.Module):
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
    dilate_scale: default 8
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
        super(PSPNet, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = norm_layer(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = norm_layer(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = norm_layer(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
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
        
        self.layer5 = self._make_psp_layer(_PyramidPoolingModule, 2048, [1, 2, 3, 6], norm_layer=norm_layer)
        #self.layer5 = self._make_psp_layer(_PyramidPoolingModule, 512*block.expansion, [1, 2, 3, 6], norm_layer=norm_layer)
            
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512*block.expansion, num_classes)
        # inplanes+len(pool_series)*reduction_planes ==  512*block.expansion+len(pool_series)*512 == 2048 + len(pool_series)*512 == 4096
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(4096, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(512, num_classes, kernel_size=1)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.001)
                
            elif isinstance(m, norm_layer):
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, norm_layer=None):
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
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation, norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unkown dilation size: {}".format(dilation))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def _make_psp_layer(self, block, inplanes, pool_series, norm_layer):
        return block(inplanes, pool_series, norm_layer)
    
    def forward(self, x):
        # x_ori = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.layer5(x)
        
        x = self.conv_last(x)
        
        """
        x_ori_size = x_ori.size()
        x_size = x.size()
        if not (x_size[2] == x_ori_size[2] and x_size[3] == x_ori_size[3]):
            x = F.interpolate(x, size=(x_ori_size[2], x_ori_size[3]), mode='bilinear') # umsample
            # x3 = nn.functional.upsample(x3, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x.size(), x_ori.size())
        """
        
        return x
        
########################################################################
# norm_layer= nn.BatchNorm2d or SynchronizedBatchNorm2d 
  
def PSPNet50(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = PSPNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    return model

def PSPNet101(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = PSPNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    return model

def PSPNet152(num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    model = PSPNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
    return model
        
if __name__ == '__main__':
    model = PSPNet50(num_classes=150)
    torch.save(model, './pretrained/pspnet50.pth')
    
    
                