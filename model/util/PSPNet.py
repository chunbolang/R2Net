import torch
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

from model.backbone.layer_extrator import layer_extrator
from torch.cuda.amp import autocast

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class OneModel(nn.Module):
    def __init__(self, args):
        super(OneModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.classes = args.base_class_num +1
        self.backbone = args.backbone

        self.fp16 = args.fp16
        
        if args.backbone in ['vgg', 'resnet50', 'resnet101']:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone, pretrained = True)
            self.encoder = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)
            fea_dim = 512 if args.backbone == 'vgg' else 2048

        # Base Learner
        bins=(1, 2, 3, 6)
        self.ppm = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim*2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, self.classes, kernel_size=1))

    def get_optim(self, model, args, LR):
        optimizer = torch.optim.SGD(
            [     
            {'params': model.encoder.parameters(), 'lr' : LR},
            {'params': model.ppm.parameters(), 'lr' : LR*10},
            {'params': model.cls.parameters(), 'lr' : LR*10},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer


    def forward(self, x, y):
        with autocast(enabled=self.fp16):
            x_size = x.size()
            h = x_size[2]
            w = x_size[3]

            x = self.encoder(x)
            if self.backbone == 'swin':
                x = self.ppm(x.permute(0, 3, 1, 2))
            else:
                x = self.ppm(x)
            x = self.cls(x)

            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

            if self.training:
                main_loss = self.criterion(x, y.long())
                return x.max(1)[1], main_loss, 0, 0
            else:
                return x