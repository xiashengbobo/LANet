#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 19:33:45 2018

@author: sunshine
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
__all__ = ['ADRNeXt', 'ADRNext50', 'ADRNext101', 'ADRNext152'] 

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve
    
model_urls = {
    'resnet18': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth',
    'resnet50': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet50-imagenet.pth',
    'resnet101': 'http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth'
    }


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class GroupBottleneck(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, groups=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(GroupBottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
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
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        #self.dropout = nn.Dropout2d(0.5)
        self.dropout = nn.Dropout2d(np.random.choice([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6], 1).item())
        
        # Attention layer
        
        # GlobalAvgPool
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        
        
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
        out = out.view(out.size(0), -1)   # --> 1D
        # print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        # print(out.size())
        out = out * original_out
        #out = original_out * out
        
        out = self.relu(out)
        
        # dropout 0.5
        out = self.dropout(out)
             
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
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
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, 
                               groups=groups, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        self.dropout = nn.Dropout2d(0.5)
        
        # Attention layer
        
        # GlobalAvgPool
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc1 = nn.Linear(in_features=planes , out_features=round(planes / 8))
        self.fc2 = nn.Linear(in_features=round(planes / 8), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        
        
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
        out = out.view(out.size(0), -1)   # --> 1D
        # print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        # print(out.size())
        out = out.view(out.size(0), out.size(1), 1, 1)
        # dropout 0.5
        out = self.dropout(out)  # change
        # print(out.size())
        out = out * original_out
        #out = original_out * out
        
        out = self.relu(out)
        
        # dropout 0.5
        # out = self.dropout(out)
             
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
    
class _AttentionModule(nn.Module):
    """
    """
    def __init__(self, inplanes_high, inplanes_low, num_classes, norm_layer=None):
        super(_AttentionModule, self).__init__()
        self.w_y = nn.Sequential(
                nn.Conv2d(inplanes_low, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                norm_layer(num_classes),
                nn.ReLU(inplace=True)
                )
        
        self.w_x = nn.Sequential(
                nn.Conv2d(inplanes_high, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
                norm_layer(num_classes),
                nn.ReLU(inplace=True)
                )
        
        self.psi = nn.Sequential(
                nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, padding=0, bias=True),
                norm_layer(1),
                nn.Sigmoid()
                )
        
    def forward(self, x_high, y_low):
        x = self.w_x(x_high)
        y = self.w_y(y_low)
        psi = self.psi(x)
        
        return y * psi + y
    
#######################################################################    
class ADRNeXt(nn.Module):
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
    
    def __init__(self, block, attentionblock, layers, num_classes, norm_layer=None, groups=32, dilate_scale=8):
        self.inplanes = 128
        super(ADRNeXt, self).__init__()
        
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
            
        #self.layer5 = self._make_psp_layer(_PyramidPoolingModule, 2048, [1, 2, 3, 6], norm_layer=norm_layer)
        self.layer5 = self._make_psp_layer(_PyramidPoolingModule, 2048, [1, 3, 6, 9], norm_layer=norm_layer)
        
        self.attention = self._make_attention_layer(_AttentionModule, 512, 128, num_classes=num_classes, norm_layer=norm_layer)
        
        self.oneplus = nn.Sequential(
                nn.Conv2d(4096, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
                )
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(512 + num_classes, 512, kernel_size=3, padding=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1)
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
                m.weight.data.normal_(0.0, 0.0001)
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
    
    def _make_attention_layer(self, block, inplanes_high, inplanes_low, num_classes, norm_layer):
        return block(inplanes_high, inplanes_low, num_classes, norm_layer)
    
    
    def forward(self, x):  # (b, 3, w, h)
        x_ori = x
        x_size = x_ori.size()
        x = self.relu1(self.bn1(self.conv1(x))) # -> (b, 64, w/2, h/2)
        x = self.relu2(self.bn2(self.conv2(x))) # -> (b, 64, w/2, h/2)
        x = self.relu3(self.bn3(self.conv3(x))) # -> (b, 64, w/2, h/2)
        x = self.maxpool(x)  # -> (b, 64, w/4, h/4)
        
        x0 = x   # -> (b, 64, w/4, h/4)
        x = self.layer1(x)    # -> (b, 256, w/4, h/4)
        x = self.layer2(x)    # -> (b, 512, w/8, h/8)
        x = self.layer3(x)    # -> (b, 1024, w/8, h/8)  or -> (b, 1024, w/16, h/16)
        x = self.layer4(x)    # -> (b, 2048, w/8, h/8)  or -> (b, 1024, w/16, h/16)
        x = self.layer5(x)    # -> (b, 4096, w/8, h/8)  or -> (b, 1024, w/16, h/16)
        x1 = self.oneplus(x)  # -> (b, 512, w/8, h/8)   or -> (b, 1024, w/16, h/16)
        
        # Upsample x2
        x0_size =x0.size()
        x1_size = x1.size()
        if not (x1_size[2] == x0_size[2] and x1_size[3] == x0_size[3]):
            x1 = nn.functional.upsample(x1, size=(x0_size[2], x0_size[3]), mode='bilinear')
        assert x1.size()[2:3] == x0.size()[2:3], "{0} vs {1}".format(x1.size(), x0.size())
        
        x_attention = self.attention(x1, x0)
        
        x1 = [x_attention, x1]
        x1 = torch.cat(x1, 1)
        
        x2 = self.conv_last(x1 )
        
        # Upsample x4
        x2_size = x2.size()
        if not (x2_size[2] == x_size[2] and x2_size[3] == x_size[3]):
            x2 = nn.functional.upsample(x2, size=(x_size[2], x_size[3]), mode='bilinear')
        assert x2.size()[2:3] == x_ori.size()[2:3], "{0} vs {1}".format(x2.size(), x_ori.size())
        
        return x2
    

###################################################################
# Models   


def ADRNext50(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ADRNeXt(GroupBottleneck, AttentionGroupBottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def ADRNext101(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ADRNeXt(GroupBottleneck, AttentionGroupBottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def ADRNext152(pretrained=False,  num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ADRNeXt(GroupBottleneck, AttentionGroupBottleneck, [3, 8, 36, 3], num_classes, norm_layer, **kwargs)
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
    model = ADRNext50(pretrained=False)
    torch.save(model, './pretrained/adrnext50.pth')