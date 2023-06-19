import random
import math

import numpy as np
import numbers
import collections
import cv2

import torch
import albumentations as Albu
from albumentations.pytorch import ToTensorV2
import util.transform as base
from torchvision import transforms as pytorch

"""
albumentations 


"""

def get_transform(transform_dict):
    """
    a dict 
    """
    pip_line = []

    if transform_dict['type'] == 'albumentations':
        tmp = 'Albu.'
    elif transform_dict['type'] == 'pytorch':
        tmp = 'pytorch.'
    else:
        tmp = 'base.'
    
    for key in transform_dict:
        if key != 'type':
            if key == 'OneOf' or key == 'SomeOf':
                tmp_pip_line = []
                for item in transform_dict[key]['transforms']:
                    tmp_pip_line.append(eval(tmp+ item.pop('type'))(**item))
                pip_line.append(eval(tmp+ key)(transforms=tmp_pip_line, p=transform_dict[key]['p']))
            elif key == 'ToTensorV2':
                pip_line.append(eval(key)())
            elif key == 'ToTensor' and tmp == 'pytorch.':
                pip_line.append(eval(tmp+ key)())
            else:
                pip_line.append(eval(tmp+ key)(**transform_dict[key]))

    transformer = eval(tmp + 'Compose')(pip_line)

    return transformer
