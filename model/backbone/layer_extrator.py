from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

import model.backbone.resnet as models
import model.backbone.vgg as vgg_models

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4


def layer_extrator(backbone = None, pretrained = True,):
    """
    
    """
    if backbone == 'vgg':
        print('INFO: Using VGG_16 bn')
        vgg_models.BatchNorm = BatchNorm
        vgg16 = vgg_models.vgg16_bn(pretrained=pretrained)
        print(vgg16)
        layer0, layer1, layer2, layer3, layer4 = get_vgg16_layer(vgg16)
    else:
        print('INFO: Using {}'.format(backbone))
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=pretrained)
        else:
            resnet = models.resnet152(pretrained=pretrained)
        layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu1, resnet.conv2, resnet.bn2, resnet.relu2, resnet.conv3, resnet.bn3, resnet.relu3, resnet.maxpool)
        layer1, layer2, layer3, layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

    return layer0, layer1, layer2, layer3, layer4