import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

from torch.cuda.amp import autocast as autocast
from model.backbone.layer_extrator import layer_extrator


class cls(nn.Module):
    def __init__(self, backbone, fp16=True):
        super(cls, self).__init__()


        self.backbone = backbone
        self.criterion = nn.CrossEntropyLoss()
        self.pretrained = True
        self.classes = 46

        self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=self.backbone, pretrained=True)
        self.fp16 = fp16

        if self.backbone == 'resnet50' or self.backbone == 'resnet101':
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * 4, self.classes )
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.classes),
        )

        self.GAP = nn.AdaptiveAvgPool2d(1)

    def get_optim(self, model, lr_dict, LR):
        optimizer = torch.optim.SGD(model.parameters(),\
            lr=LR, momentum=lr_dict['momentum'], weight_decay=lr_dict['weight_decay'])
        
        return optimizer
    
    def forward(self, x,  y):
        with autocast(enabled=self.fp16):
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            # x = F.log_softmax(x, 1,)
            if self.training:
                loss = self.criterion(x, y.long())
                # loss_1 = F.nll_loss(x,y.long())
                return x, loss
            else:
                return x

    #   
    