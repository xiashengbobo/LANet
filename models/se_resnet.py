#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 10:49:11 2018

@author: sunshine
"""

# SENet - based resnet with dilation

import os
import sys
import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from lib.nn.modules.batchnorm import SynchronizedBatchNorm2d


# 提供了暴露接口用的“白名单”
__all__ = ['SENet', 'SE_resnet18', 'SE_resnet50', 'SE_resnet101', 'SE_resnet152'] 

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


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.downsample = downsample
        self.stride = stride
        
        # Attention layer
        
        # GlobalAvgPool
        """
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        """
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
        self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
        self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # Attention
        original_out = out
        print(out.size())
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1) # --> 1D
        print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
        
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
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # Attention layer
        
        # GlobalAvgPool
        """
        if planes == 64:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=56, stride=1)
        elif planes == 128:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=28, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=14, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        """
        
        self.globalAvgPool = nn.AdaptiveAvgPool2d(output_size=(1,1))
        
        self.fc1 = nn.Linear(in_features=planes * 4, out_features=round(planes / 4))
        self.fc2 = nn.Linear(in_features=round(planes / 4), out_features=planes * 4)
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
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
         # Attention
        original_out = out
        print(out.size())
        out = self.globalAvgPool(out)
        out = out.view(out.size(0), -1)   # --> 1D
        print(out.size())
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        out = out * original_out
        
        out += residual
        out = self.relu(out)
        
        return out
    
class GroupBottleneck_v2(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(GroupBottleneck_v2, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # Attention layer
        
        # GlobalAvgPool
        """
        if planes == 128:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=56, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=28, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=14, stride=1)
        elif planes == 1024:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        """
        
        # Attention layer
        # Sub-region
        self.subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3/5/7
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes, affine=affine_par),
                nn.ReLU(inplace=True),
                )
        
        self.fc = nn.Sequential(
                nn.Linear(in_features=planes, out_features=int(planes // 16), bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=int(planes // 16), out_features=planes, bias=True),
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
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        # Attention
        original_out = out
        # print(out.size())
        out = self.subAvgPool(out)
        out = out.view(out.size(0), -1)   # --> 1D (b, c)
        # print(out.size())
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1)  # (b, c, 1, 1)
        # print(out.size())
        
        # out = out * original_out
        out = original_out * out
        
        out += residual
        out = self.relu(out)
        
        return out
    
class GroupBottleneck_v3(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(GroupBottleneck_v3, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # Attention layer
        
        # GlobalAvgPool
        """
        if planes == 128:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=56, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=28, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=14, stride=1)
        elif planes == 1024:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        """
        
        # Attention layer
        # Sub-region
        self.subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool 3/5/7
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes, affine=affine_par),
                nn.ReLU(inplace=True),
                )
        
        self.fc = nn.Sequential(
                nn.Linear(in_features=planes, out_features=int(planes // 16), bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(in_features=int(planes // 16), out_features=planes, bias=True),
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
        out = self.subAvgPool(out)
        out = out.view(out.size(0), -1)   # --> 1D
        # print(out.size())
        out = self.fc(out)
        out = out.view(out.size(0), out.size(1), 1, 1)
        # print(out.size())
        
        # out = out * original_out
        out = original_out * out
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
    
class GroupBottleneck_v4(nn.Module):
    """
    ResNet Bottleneck
    """
    expansion = 2
    
    def __init__(self, inplanes, planes, stride=1, dilation=1, 
                 downsample=None, previous_dilation=1, norm_layer=None):
        super(GroupBottleneck_v4, self).__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        
        #padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation = dilation, bias=False)
        self.bn2 = norm_layer(planes)
        
        self.conv3 = nn.Conv2d(planes, planes*2, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * 2)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride
        
        # Attention layer
        
        # GlobalAvgPool
        """
        if planes == 128:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=56, stride=1)
        elif planes == 256:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=28, stride=1)
        elif planes == 512:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=14, stride=1)
        elif planes == 1024:
            self.globalAvgPool = nn.AvgPool2d(kernel_size=7, stride=1)
        """
        
        # Attention layer
        # Sub-region
        self.subAvgpool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(3,3)),  # GlobalAvgPool
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes, affine=affine_par),
                nn.ReLU(inplace=True),
                # nn.Dropout2d(0.2),
                nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, padding=0, dilation=1, bias=True),
                norm_layer(planes, affine=affine_par),
                nn.Sigmoid()  # nn.Softmax(dim=1) or nn.Sigmoid()
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
        out = self.subAvgPool(out)
        
        # out = out * original_out
        out = original_out * out
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out
    
    
class SENet(nn.Module):
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
    
    def __init__(self, block, layers, num_classes, norm_layer=None, dilate_scale=8):
        self.inplanes = 64
        super(SENet, self).__init__()
        
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
            
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512*block.expansion, num_classes)
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(512*block.expansion, 512, kernel_size=3, padding=1, 
                          bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(512, num_classes, kernel_size=1)
                )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                """
                if m.bias is not None:
                    m.bias.data.zero_()
                """
                    
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                
                # m.weight.data.normal_(0, 0.001)
                
            elif isinstance(m, norm_layer):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
                # m.weight.data.normal_(1.0, 0.02)
                # m.bias.data.fill_(0) # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, 
                    norm_layer=None):
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
                                downsample=downsample, previous_dilation=dilation, 
                                norm_layer=norm_layer))
        elif dilation == 4:
            layers.append(block(self.inplanes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation,
                                norm_layer=norm_layer))
        else:
            raise RuntimeError("=> unkown dilation size: {}".format(dilation))
        
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.conv_last(x)
        
        return x
    

###################################################################
# Models   

def SE_resnet18(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def SE_resnet34(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(BasicBlock, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def SE_resnet50(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet50']), strict=False)
    return model


def SE_resnet101(pretrained=False, num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(model_urls['resnet101']), strict=False)
    return model


def SE_resnet152(pretrained=False,  num_classes=150, norm_layer=nn.BatchNorm2d, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SENet(Bottleneck, [3, 8, 36, 3], **kwargs)
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
    model = SE_resnet50(pretrained=True)