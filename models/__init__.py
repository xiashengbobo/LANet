#

# import torchvision.models as models

from models.resnet import resnet18, resnet50, resnet101
from models.resnext import resnext50, resnext101
from models.dilated_resnet import dilated_resnet18, dilated_resnet50, dilated_resnet101, dilated_resnet152
from models.se_resnet import SE_resnet18, SE_resnet50, SE_resnet101, SE_resnet152
from models.se_resnext import SE_resnext50, SE_resnext101, SE_resnext152
from models.deeplab import DeepLab50, DeepLab101, DeepLab152
from models.deeplabv3 import DeepLabv3_50, DeepLabv3_101, DeepLabv3_152
from models.pspnet import PSPNet50, PSPNet101, PSPNet152

# Our networks
from models.adrnext import ADRNext50, ADRNext101, ADRNext152
from models.atdnext import ATDNext50, ATDNext101, ATDNext152
from models.amdnext import AMDNext50, AMDNext101, AMDNext152
from models.aspnet import ASPNet50, ASPNet101, ASPNet152, ASPNet50_v1, ASPNet101_v1, ASPNet152_v1, ASPNet50_v2, ASPNet101_v2, ASPNet152_v2
from models.daspnet import DASPNet50, DASPNet101, DASPNet152, DASPNet50_v2, DASPNet101_v2, DASPNet152_v2
from models.paspnet import PASPNet50, PASPNet101, PASPNet152
from models.caspnet import CASPNet50, CASPNet101, CASPNet152


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
                "DeepLabv3_50" : DeepLabv3_50,
                "DeepLabv3_101" : DeepLabv3_101,
                "DeepLabv3_152" : DeepLabv3_152,
                "ADRNext50" : ADRNext50,
                "ADRNext101" : ADRNext101,
                "ADRNext152" : ADRNext152,
                "ATDNext50" : ATDNext50,
                "ATDNext101" : ATDNext101,
                "ATDNext152" : ATDNext152,
                "AMDNext50" : AMDNext50,
                "AMDNext101" : AMDNext101,
                "AMDNext152" : AMDNext152,
                "ASPNet50" : ASPNet50,
                "ASPNet101" : ASPNet101,
                "ASPNet152" : ASPNet152,
                "ASPNet50_v1" : ASPNet50_v1,
                "ASPNet101_v1" : ASPNet101_v1,
                "ASPNet152_v1" : ASPNet152_v1,
                "ASPNet50_v2" : ASPNet50_v2,
                "ASPNet101_v2" : ASPNet101_v2,
                "ASPNet152_v2" : ASPNet152_v2,
                "DASPNet50" : DASPNet50,
                "DASPNet101" : DASPNet101,
                "DASPNet152" : DASPNet152,
                "DASPNet50_v2" : DASPNet50_v2,
                "DASPNet101_v2" : DASPNet101_v2,
                "DASPNet152_v2" : DASPNet152_v2,
                "PASPNet50" : PASPNet50,
                "PASPNet101" : PASPNet101,
                "PASPNet152" : PASPNet152,
                "CASPNet50" : CASPNet50,
                "CASPNet101" : CASPNet101,
                "CASPNet152" : CASPNet152,
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
    elif name == 'DeepLabv3_50':
        model = model(num_classes = num_classes)
    elif name == 'DeepLabv3_101':
        model = model(num_classes = num_classes)
    elif name == 'DeepLabv3_152':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext50':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext101':
        model = model(num_classes = num_classes)
    elif name == 'ADRNext152':
        model = model(num_classes = num_classes)
    elif name == 'ATDNext50':
        model = model(num_classes = num_classes)
    elif name == 'ATDNext101':
        model = model(num_classes = num_classes)
    elif name == 'ATDNext152':
        model = model(num_classes = num_classes)
    elif name == 'AMDNext50':
        model = model(num_classes = num_classes)
    elif name == 'AMDNext101':
        model = model(num_classes = num_classes)
    elif name == 'AMDNext152':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet50':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet101':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet152':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet50_v1':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet101_v1':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet152_v1':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet50_v2':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet101_v2':
        model = model(num_classes = num_classes)
    elif name == 'ASPNet152_v2':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet50':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet101':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet152':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet50_v2':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet101_v2':
        model = model(num_classes = num_classes)
    elif name == 'DASPNet152_v2':
        model = model(num_classes = num_classes)
    elif name == 'PASPNet50':
        model = model(num_classes = num_classes)
    elif name == 'PASPNet101':
        model = model(num_classes = num_classes)
    elif name == 'PASPNet152':
        model = model(num_classes = num_classes)
    elif name == 'CASPNet50':
        model = model(num_classes = num_classes)
    elif name == 'CASPNet101':
        model = model(num_classes = num_classes)
    elif name == 'CASPNet152':
        model = model(num_classes = num_classes)
    else:
        model = model(num_classes = num_classes)
    return model