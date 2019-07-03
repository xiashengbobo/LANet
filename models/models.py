#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 13:59:46 2018

@author: sunshine
"""


#import copy
import torchvision.models as models

from resnet import resnet18, resnet50, resnet101
from resnext import resnext50, resnext101
from dilated_resnet import dilated_resnet18, dilated_resnet50, dilated_resnet101, dilated_resnet152
from se_resnet import SE_resnet18, SE_resnet50, SE_resnet101, SE_resnet152
from se_resnext import SE_resnext50, SE_resnext101, SE_resnext152
from deeplab import DeepLab50, DeepLab101, DeepLab152
from pspnet import PSPNet50, PSPNet101, PSPNet152

from adrnext import ADRNext50, ADRNext101, ADRNext152

def _get_model_instance(name):
    try:
        return{
                "resnet18" : resnet18,
                "resnet50" : resnet50,
                "resnet101" : resnet101,
                "resnext50" : resnext50,
                "resnext101" : resnext101,
                "dilated_resnet18" : dilated_resnet18,
                "dilated_resnet50" : dilated_resnet50,
                "dilated_resnet101" : dilated_resnet101,
                "dilated_resnet152" : dilated_resnet152,
                "SE_resnet18" : SE_resnet18,
                "SE_resnet50" : SE_resnet50,
                "SE_resnet101" : SE_resnet101,
                "SE_resnet152" : SE_resnet152,
                "SE_resnext50" : SE_resnext50,
                "SE_resnext101" : SE_resnext101,
                "SE_resnext152" : SE_resnext152,
                "PSPNet50" : PSPNet50,
                "PSPNet101" : PSPNet101,
                "PSPNet152" : PSPNet152,
                "DeepLab50" : DeepLab50,
                "DeepLab101" : DeepLab101,
                "DeepLab152" : DeepLab152, 
                "ADRNext50" : ADRNext50,
                "ADRNext101" : ADRNext101,
                "ADRNext152" : ADRNext152,
                }[name]
    except:
        print('Model {} not available'.format(name))
            
def get_model(name, num_classes, version=None):
    model = _get_model_instance(name)
    
    if name == 'resnet18':
        model = model(num_classes = num_classes)
    elif name == 'resnet50':
        model = model(num_classes = num_classes)
    elif name == 'resnet101':
        model = model(num_classes = num_classes)
    elif name == 'resnext50':
        model = model(num_classes = num_classes)
    elif name == 'resnext101':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnet18':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnet50':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnet101':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnet152':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnext50':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnext101':
        model = model(num_classes = num_classes)
    elif name == 'SE_resnext152':
        model = model(num_classes = num_classes)
    elif name == 'PSPNet50':
        model = model(num_classes = num_classes)
    elif name == 'PSPNet101':
        model = model(num_classes = num_classes)
    elif name == 'PSPNet152':
        model = model(num_classes = num_classes)
    elif name == 'DeepLab50':
        model = model(num_classes = num_classes)
    elif name == 'DeepLab101':
        model = model(num_classes = num_classes)
    elif name == 'DeepLab152':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext50':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext101':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext152':
        model = model(num_classes = num_classes)
    else:
        model = model(num_classes = num_classes)
    return model
        
if __name__ == '__main__':
    model = get_model(name='ADRNext50', num_classes=150)
    #for name, param in model.named_parameters():
        #print(name, param.size())
    
    
    
    
    
    
    
    
    
    
    
    