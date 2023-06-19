import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

from model.backbone.layer_extrator import layer_extrator

from model.util.ASPP import ASPP, ASPP_Drop ,ASPP_BN
from model.util.PSPNet import OneModel as PSPNet
from torch.cuda.amp import autocast as autocast

def Cor_Map(query_feat, supp_feat_list, mask_list):
    corr_query_mask_list = []
    cosine_eps = 1e-7
    for i, tmp_supp_feat in enumerate(supp_feat_list):
        resize_size = tmp_supp_feat.size(2)
        tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

        tmp_supp_feat_4 = tmp_supp_feat * tmp_mask
        q = query_feat
        s = tmp_supp_feat_4
        bsize, ch_sz, sp_sz, _ = q.size()[:]

        tmp_query = q
        tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
        tmp_query_norm = torch.norm(tmp_query, 2, 1, True)

        tmp_supp = s               
        tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
        tmp_supp = tmp_supp.contiguous().permute(0, 2, 1)
        tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

        similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
        similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)
        similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
        corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
        corr_query_mask_list.append(corr_query)  
    corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)
    return corr_query_mask

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def pro_select(query_feat, pro_list):
    """
    query_feat  b*c*h*w
    pro_list  [b*1*c*1*1]*n  
    """
    query_base = query_feat.unsqueeze(1) # b*1*c*h*w
    pro_gather = torch.cat(pro_list, 1)  # b*n*c*1*1
    pro_gather = pro_gather.expand(-1,-1,-1,query_base.size(3), query_base.size(4)) # b*n*c*h*w

    index_map = nn.CosineSimilarity(2)(pro_gather, query_base).unsqueeze(2) # b*n*1*h*w
    index_pro = index_map.max(1)[1].unsqueeze(1).expand_as(query_base)  # b*1*c*h*w
    out_pro = torch.gather(pro_gather, 1, index_pro).squeeze(1)
    out_map = torch.sum(index_map, 1)

    return out_pro, out_map

class feat_decode(nn.Module):
    def __init__(self, inchannel):
        super(feat_decode, self).__init__()

        self.ASPP = ASPP()

        self.res1 = nn.Sequential(
            nn.Conv2d(inchannel*5, inchannel, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )
        
        self.res2 = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )    

        self.cls = nn.Sequential(
            nn.Conv2d(inchannel, inchannel, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(inchannel, 1, kernel_size=1)
        )

    def forward(self,x):
        x =self.ASPP(x)
        x = self.res1(x)
        x = self.res2(x) + x
        x = self.cls(x)

        return x

class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()


        self.shot = args.shot
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.pretrained = args.pretrain
        self.classes = 2
        self.fp16 = args.fp16
        self.backbone = args.backbone
        self.base_class_num = args.base_class_num

        self.alpha = torch.nn.Parameter(torch.FloatTensor(self.base_class_num+1,1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.FloatTensor(self.base_class_num+1,1), requires_grad=True)
        nn.init.normal_(self.alpha, mean=1.0)
        nn.init.normal_(self.beta)
        
        self.pro_global = torch.FloatTensor(self.base_class_num+1, 256).cuda()
        nn.init.normal_(self.pro_global)
        # self.pro_global.data.fill_(0.0)

        if self.pretrained:
            BaseNet = PSPNet(args)
            weight_path = 'initmodel/PSPNet/{}/split{}/{}/best.pth'.format(args.dataset, args.split, args.backbone)
            new_param = torch.load(weight_path, map_location=torch.device('cpu'))['state_dict']
            print('load <base> weights from: {}'.format(weight_path))
            try: 
                BaseNet.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                BaseNet.load_state_dict(new_param)
            
            self.layer0, self.layer1, self.layer2, \
                self.layer3, self.layer4 = BaseNet.layer0, BaseNet.layer1, BaseNet.layer2, BaseNet.layer3, BaseNet.layer4

            self.base_layer = nn.Sequential(BaseNet.ppm, BaseNet.cls)
        else:
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = layer_extrator(backbone=args.backbone, pretrained=True)

        reduce_dim = 256
        if self.backbone == 'vgg':
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.init_merge = nn.Sequential(
            nn.Conv2d(reduce_dim*2 +  1 , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))


        self.init_merge_bg = nn.Sequential(
            nn.Conv2d(reduce_dim*2 , reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True))


        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.decode = feat_decode(inchannel= reduce_dim)
        
        self.iter = 0

    def get_optim(self, model, lr_dict, LR):
        optimizer = torch.optim.SGD(
            [
            {'params': model.alpha},
            {'params': model.beta},
            {'params': model.pro_global},
            {'params': model.down_query.parameters()},
            {'params': model.down_supp.parameters()},
            {'params': model.init_merge.parameters()},
            {'params': model.init_merge_bg.parameters()},
            {'params': model.decode.parameters()},
            ],

            lr=LR, momentum=lr_dict['momentum'], weight_decay=lr_dict['weight_decay'])
        
        return optimizer


    def forward(self, x, s_x, s_y, y, cat_idx=None):
        with autocast(enabled=self.fp16):
            x_size = x.size()
            h = x_size[2]
            w = x_size[3]

            # Query Feature
            with torch.no_grad():
                query_feat_0 = self.layer0(x)
                query_feat_1 = self.layer1(query_feat_0)
                query_feat_2 = self.layer2(query_feat_1)
                query_feat_3 = self.layer3(query_feat_2)
                query_feat_4 = self.layer4(query_feat_3)
                query_out = self.base_layer(query_feat_4)
                query_out = nn.Softmax2d()(query_out)
            
                if self.backbone == 'vgg':
                    query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

            query_feat = torch.cat([query_feat_3, query_feat_2], 1)
            query_feat = self.down_query(query_feat)
            
            no_base_map = self.get_no_base_map(query_out, cat_idx)

            # Support Feature     
            final_supp_list = []
            mask_list = []
            act_fg_list = []
            act_bg_list = []
            feat_fg_list = []
            feat_bg_list = []
            aux_loss_1 = 0
            aux_loss_2 = 0
            for i in range(self.shot):
                pro_fg_list = []
                pro_bg_list = []
                mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
                mask_list.append(mask)
                with torch.no_grad():
                    supp_feat_0 = self.layer0(s_x[:,i,:,:,:])
                    supp_feat_1 = self.layer1(supp_feat_0)
                    supp_feat_2 = self.layer2(supp_feat_1)
                    supp_feat_3 = self.layer3(supp_feat_2)
                    supp_feat_4_true = self.layer4(supp_feat_3)

                    mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                    supp_feat_4 = self.layer4(supp_feat_3*mask)
                    final_supp_list.append(supp_feat_4)
                    if self.backbone == 'vgg':
                        supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                
                    supp_base_out = self.base_layer(supp_feat_4_true.clone())
                    supp_base_out = nn.Softmax2d()(supp_base_out) # b*(c+1)*h*w
                    

                supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
                supp_feat_tmp = self.down_supp(supp_feat)

                # gen pro
                pro_fg = Weighted_GAP(supp_feat_tmp , mask)
                pro_fg_list.append(pro_fg.unsqueeze(1) )
                pro_bg = Weighted_GAP(supp_feat_tmp , 1-mask)
                pro_bg_list.append(pro_bg.unsqueeze(1))
                # gen pro by act_map
                pro_list = []
                for j in range(self.base_class_num +1 ):
                    pro = Weighted_GAP(supp_feat_tmp , supp_base_out[:,j,].unsqueeze(1)).unsqueeze(1) # b*1*256*1*1
                    pro_list.append(pro)
                    if not self.training and j ==0 :
                        pro_fg_list.append(pro)

                local_pro = torch.cat(pro_list, 1)  #b*(c+1)*256*1*1

                # global 2 local

                cur_local_pro = ((self.alpha * self.pro_global).unsqueeze(0) + (self.beta).unsqueeze(0) * local_pro.squeeze(3).squeeze(3)).unsqueeze(3).unsqueeze(3) # b*(c+1)*256*1*1
                # local 2 global
                # with torch.no_grad():
                # new_pro_global = self.pro_global * self.beta + torch.mean(local_pro, 0).squeeze(2).squeeze(2) * (1-self.beta)
                # with torch.no_grad():
                #     self.pro_global = new_pro_global.clone()

                base_fg_list = []
                base_bg_list = []

                # select pro
                for b_id in range(query_feat.size(0)):
                    c_id_array = torch.arange(self.base_class_num+1, device='cuda')
                    c_id = cat_idx[0][b_id] + 1
                    c_mask = (c_id_array!=c_id)

                    if self.training:
                        base_fg_list.append(cur_local_pro[b_id, c_id,:,: ].unsqueeze(0))  # b*256*1*1
                        base_bg_list.append(cur_local_pro[b_id,c_mask,:,:].unsqueeze(0)) # b*c*256*1*1
                    else:
                        base_bg_list.append(cur_local_pro[b_id,:,:,:].unsqueeze(0))  # b*(c+1)*1*1

                if self.training:
                    base_fg = torch.cat(base_fg_list, 0)  # b*1*256*1*1
                    base_bg = torch.cat(base_bg_list, 0)  # b*c(c+1)*256*1*1
                    pro_fg_list.append(base_fg.unsqueeze(1))
                    pro_bg_list.append(base_bg)
                    tmp_pro = torch.mean(cur_local_pro, 0 ).squeeze(2)
                    # tmp_pro = self.pro_global.unsqueeze(2) # (c+1)*256*1
                    crs = nn.CosineSimilarity(1)(tmp_pro, tmp_pro.transpose(0,2))  # (c+1)*(c+1)
                    crs[crs==1] = 0
                    crs_1 = 1-nn.CosineSimilarity(1)(torch.mean(local_pro, 0).squeeze(2), tmp_pro)
                    gamma = (self.base_class_num+1) * self.base_class_num  
                    aux_loss_1 +=  (torch.sum(crs) / gamma )
                    aux_loss_2 += torch.mean(crs_1)
                else:
                    base_bg = torch.cat(base_bg_list, 0)  # b*c(c+1)*256*1*1
                    pro_bg_list.append(base_bg)
                
                fg_feat, fg_map = pro_select(query_feat, pro_fg_list)
                bg_feat, bg_map = pro_select(query_feat, pro_bg_list)
                
                feat_bg_list.append(bg_feat.unsqueeze(1))
                feat_fg_list.append(fg_feat.unsqueeze(1))
                act_bg_list.append(bg_map.unsqueeze(1))
                act_fg_list.append(fg_map.unsqueeze(1))
                
            if self.shot>1:
                aux_loss_1 /= self.shot
                aux_loss_2 /= self.shot
                fg_dis = torch.cat(act_fg_list, 1)
                bg_dis = torch.cat(act_bg_list, 1)  # b*k*1*h*w
                fg_dis =F.softmax(fg_dis, 1)
                bg_dis =F.softmax(bg_dis, 1)
                fg_feat = torch.mean(torch.cat(feat_fg_list, 1)*fg_dis, 1)
                bg_feat = torch.mean(torch.cat(feat_bg_list, 1)*bg_dis, 1)
                # bg_map = torch.mean(torch.cat(act_bg_list, 1), 1)
                # fg_map = torch.mean(torch.cat(act_fg_list, 1), 1)
                

            corr_query_mask = Cor_Map(query_feat_4, final_supp_list, mask_list )
            corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)

            query_feat_bin = query_feat

            merge_feat_bin = torch.cat([query_feat_bin, fg_feat, corr_query_mask], 1)
            merge_feat_bin = self.init_merge(merge_feat_bin)     

            merge_feat_bg_bin = torch.cat([query_feat_bin, bg_feat], 1)
            merge_feat_bg_bin = self.init_merge_bg(merge_feat_bg_bin)   

            out_fg = self.decode(merge_feat_bin * no_base_map)
            out_bg = self.decode(merge_feat_bg_bin)
            output_fin = torch.cat([out_bg, out_fg], 1)

            #   Output Part
            output_fin = F.interpolate(output_fin, size=(h, w), mode='bilinear', align_corners=True)
                
            if self.training:
                act_map = nn.Softmax(1)(output_fin)
                alpha = self.GAP(act_map[:,1].unsqueeze(1))
                main_loss = self.criterion(output_fin, y.long()) 

                mask_y = (y==1).float().unsqueeze(1)
                alpha_1 = self.GAP(mask_y)
                beta = (alpha - alpha_1)**2
                
                aux_loss = -(1-alpha)*torch.log(alpha) - beta * torch.log(1-beta)
                return output_fin.max(1)[1], main_loss, torch.mean(aux_loss), aux_loss_1 + aux_loss_2
            else:
                return output_fin

    def get_no_base_map(self, query_out, cat_idx):
        map_list = []
        if self.training:
            for b_id in range(query_out.size(0)):
                c_id = cat_idx[0][b_id] + 1
                current_map = query_out[b_id,c_id,]
                bg_map = query_out[b_id,0,]
                map_list.append((bg_map + current_map).unsqueeze(0))
                
            out_map = torch.cat(map_list, 0).unsqueeze(1)
        else:
            out_map = query_out[:,0,]
        
        return out_map