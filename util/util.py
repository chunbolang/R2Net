import os
import numpy as np
from PIL import Image
import random
import logging
import cv2
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib.ticker import FuncFormatter, FormatStrFormatter
from matplotlib import font_manager
from matplotlib import rcParams
import seaborn as sns
import pandas as pd
import math
from seaborn.distributions import distplot
from tqdm import tqdm
from scipy import ndimage

# from get_weak_anns import find_bbox, ScribblesRobot

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.init as initer

Special_characters = [
    ['▁▂▃▄▅▆▇█', '█▇▆▅▄▃▂▁'],
    ['( * ^ _ ^ * )', '( * ^ _ ^ * )'],
    ['( $ _ $ )', '( $ _ $ )'],
    ['(？ ‧ _ ‧ )', '( ‧ _ ‧ ？)'],
    ['( T___T )', '( T___T )'],
    [' ⌒ _ ⌒ ☆ ', ' ☆ ⌒ _ ⌒ '],
    ['( = ^ o ^ = )', '( = ^ o ^ = )'],
    [' ㊣ ㊣ ㊣ ', ' ㊣ ㊣ ㊣ '],
    ['.¸.·´¯`·', '.¸.·´¯`·'],
    ['( ¯ □ ¯ )', '( ¯ □ ¯ )'],
    ['( ⊙ o ⊙ )', '( ⊙ o ⊙ )'],
    [' ◕ ‿ ◕ ｡ ', ' ｡ ◕ ‿ ◕ '],
    ['( ◡ ‿ ◡ ✿)', '(✿ ◡ ‿ ◡ )']
]

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def lr_decay(optimizer, base_lr, curr_iter, max_iter, decay_dict, current_characters=None):
    if decay_dict['type'] == 'poly_learning_rate':
        scale_lr=10.
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** decay_dict['power']
        for index, param_group in enumerate(optimizer.param_groups):
            if index <= decay_dict['index_split']:
                param_group['lr'] = lr
            else:
                param_group['lr'] = lr * scale_lr

    elif decay_dict['type'] == 'adjust_learning_rate_poly':
        reduce = ((1-float(curr_iter)/max_iter)**(decay_dict['power']))
        lr = base_lr * reduce
        optimizer.param_groups[0]['lr'] = lr * 1
        optimizer.param_groups[1]['lr'] = lr * 2
        optimizer.param_groups[2]['lr'] = lr * 10
        optimizer.param_groups[3]['lr'] = lr * 20
    
    elif decay_dict['type'] == 'half_learning_rate':
        scale_lr=10.
        lr = base_lr * (1 - float(decay_dict['rate']*curr_iter) / (max_iter)) ** decay_dict['power']

        for index, param_group in enumerate(optimizer.param_groups):
            if index <= decay_dict['index_split']:
                param_group['lr'] = lr
            else:
                param_group['lr'] = lr * scale_lr

    elif decay_dict['type'] == 'split_learning_rate_poly':
        reduce = ((1-float(curr_iter)/max_iter)**(decay_dict['power']))
        lr = base_lr * reduce
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr * decay_dict['scale_lr'][i]

    elif decay_dict['type'] == 'step_learning_rate_poly':
        # reduce = ((1-float(curr_iter)/max_iter)**(decay_dict['power']))
        
        # tmp_lr =base_lr
        for i in len(decay_dict['step']):

            if float(curr_iter)/max_iter > decay_dict['step'][i]:
                tmp_lr = base_lr*decay_dict['step_rate'][i]
        # lr = tmp_lr * reduce

        for param_group in optimizer.param_groups:
            param_group['lr'] = tmp_lr
    elif decay_dict['type'] == 'adam':
        lr = base_lr
        if curr_iter % 50 == 0:   
            print(' Using Adam optim')

        
    if curr_iter % 50 == 0:   
        print(' '*len(current_characters[0])*3 +' '*10  + 'Base LR: {:.8f}, Curr LR: {:.8f}'.format(base_lr, lr) )

def step_learning_rate(optimizer, base_lr, epoch, step_epoch, multiplier=0.1):
    """Sets the learning rate to the base LR decayed by 10 every step epochs"""
    lr = base_lr * (multiplier ** (epoch // step_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def poly_learning_rate(optimizer, base_lr, curr_iter, max_iter, power=0.9, index_split=-1, scale_lr=10., warmup=False, warmup_step=500):
    """poly learning rate policy"""
    if warmup and curr_iter < warmup_step:
        lr = base_lr * (0.1 + 0.9 * (curr_iter/warmup_step))
    else:
        lr = base_lr * (1 - float(curr_iter) / max_iter) ** power

    if curr_iter % 50 == 0:   
        print('Base LR: {:.4f}, Curr LR: {:.4f}, Warmup: {}.'.format(base_lr, lr, (warmup and curr_iter < warmup_step)))     

    for index, param_group in enumerate(optimizer.param_groups):
        if index <= index_split:
            param_group['lr'] = lr
        else:
            param_group['lr'] = lr * scale_lr

def adjust_learning_rate_poly(optimizer, base_lr, curr_iter, max_iter,  power=0.9):
    # base_lr = 3.5e-4
    # max_iter = args.max_steps
    reduce = ((1-float(curr_iter)/max_iter)**(power))
    lr = base_lr * reduce
    optimizer.param_groups[0]['lr'] = lr * 1
    optimizer.param_groups[1]['lr'] = lr * 2
    optimizer.param_groups[2]['lr'] = lr * 10
    optimizer.param_groups[3]['lr'] = lr * 20
    if curr_iter % 50 == 0:   
        print('Base LR: {:.8f}, Curr LR: {:.8f}, '.format(base_lr, lr)) 

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    # target = target.view(-1)
    target = target.reshape(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def del_file(path):
    for i in os.listdir(path):
        path_file = os.path.join(path,i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            del_file(path_file)

def init_weights(model, conv='kaiming', batchnorm='normal', linear='kaiming', lstm='kaiming'):
    """
    :param model: Pytorch Model which is nn.Module
    :param conv:  'kaiming' or 'xavier'
    :param batchnorm: 'normal' or 'constant'
    :param linear: 'kaiming' or 'xavier'
    :param lstm: 'kaiming' or 'xavier'
    """
    for m in model.modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if conv == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif conv == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of conv error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):#, BatchNorm1d, BatchNorm2d, BatchNorm3d)):
            if batchnorm == 'normal':
                initer.normal_(m.weight, 1.0, 0.02)
            elif batchnorm == 'constant':
                initer.constant_(m.weight, 1.0)
            else:
                raise ValueError("init type of batchnorm error.\n")
            initer.constant_(m.bias, 0.0)

        elif isinstance(m, nn.Linear):
            if linear == 'kaiming':
                initer.kaiming_normal_(m.weight)
            elif linear == 'xavier':
                initer.xavier_normal_(m.weight)
            else:
                raise ValueError("init type of linear error.\n")
            if m.bias is not None:
                initer.constant_(m.bias, 0)

        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight' in name:
                    if lstm == 'kaiming':
                        initer.kaiming_normal_(param)
                    elif lstm == 'xavier':
                        initer.xavier_normal_(param)
                    else:
                        raise ValueError("init type of lstm error.\n")
                elif 'bias' in name:
                    initer.constant_(param, 0)

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color

# ------------------------------------------------------
def get_model_para_number(model):
    total_number = 0
    learnable_number = 0 
    for para in model.parameters():
        total_number += torch.numel(para)
        if para.requires_grad == True:
            learnable_number+= torch.numel(para)
    return total_number, learnable_number

def setup_seed(seed=2021, deterministic=False):
    if deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

def get_metirc(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    #交并比
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    if intersection.shape[0] == 0:
        area_intersection = torch.tensor([0.,0.],device='cuda')
    else:
        area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    # area_union = area_output + area_target - area_intersection
    Pre = area_intersection / (area_output + 1e-10)
    Rec = area_intersection / (area_target + 1e-10)
    return Pre, Rec

def get_save_path_1(args):
    if len(args.variable1) != 0 :
        variable1 = eval('args.{}'.format(args.variable1))
        args[args.variable1] = eval(args.aux1)
    else :
        variable1 = 0
    if len(args.variable2) != 0 :
        variable2 = eval('args.{}'.format(args.variable2))
        args[args.variable2] = eval(args.aux2)
    else :
        variable2 = 0

    args.snapshot_path = 'exp/{}/{}/{}/split{}/{}shot/'.format( args.arch,  args.dataset, args.backbone, args.split, args.shot)
    args.result_path = 'exp/{}/{}/{}/split{}/result/'.format( args.arch,  args.dataset, args.backbone, args.split)
    if len(args.variable1) != 0 or len(args.variable2) != 0  :
        args.snapshot_path = args.snapshot_path+'{}_{}/'.format(eval(args.aux1), eval(args.aux2))
        
def get_save_path(args):
    if len(args.variable1) != 0 :
        variable1 = eval('args.{}'.format(args.variable1))

    else :
        variable1 = 0
    if len(args.variable2) != 0 :
        variable2 = eval('args.{}'.format(args.variable2))

    else :
        variable2 = 0

    args.snapshot_path = 'exp/{}/{}/{}/split{}/{}shot/'.format( args.arch,  args.dataset, args.backbone, args.split, args.shot)
    args.result_path = 'exp/{}/{}/{}/split{}/result/'.format( args.arch,  args.dataset, args.backbone, args.split)
    if len(args.variable1) != 0 or len(args.variable2) != 0  :
        args.snapshot_path = args.snapshot_path+'{}_{}/'.format(variable1, variable2)
        

def is_same_model(model1, model2):
    flag = 0
    count = 0
    for k, v in model1.state_dict().items():
        model1_val = v
        model2_val = model2.state_dict()[k]
        if (model1_val==model2_val).all():
            pass
        else:
            flag+=1
            print('value of key <{}> mismatch'.format(k))
        count+=1

    return True if flag==0 else False

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
    if classname.find('LayerNorm') != -1:
        m.eval()

def freeze_modules(model, freeze_layer):

    for item in freeze_layer:
        if hasattr(model, item):
            for param in eval('model.' + item).parameters():
                param.requires_grad = False
            print('model.{} has been frozen'.format(item))
        else:
            print('model has no part named {} , please check the input'.format(item))

def sum_list(list):
    sum = 0
    for item in list:
        sum += item
    return sum

def make_dataset(data_root=None, data_list=None, all_class=None, split_list=None, fliter_intersection=True):    
    # if not os.path.isfile(data_list):
    #     raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    flitered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2, 
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32    
    image_label_list_0 = []  
    image_label_list_1 = [] 
    image_label_list_2 = [] 
    image_label_list_3 = [] 
    fin_list = []
    list_read = open(data_list).readlines()

    sub_class_file_list = {}
    for sub_c in all_class:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        # all_label_list = []
        new_label_class_0 = []
        new_label_class_1 = []
        new_label_class_2 = []
        new_label_class_3 = []
        all_label_class = []
        
        if fliter_intersection:

            for c in label_class:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0],target_pix[1]] = 1 
                
                if tmp_label.sum() >= 2 * 32 * 32:   
                    all_label_class.append(c)
                    for i in range(len(split_list)):
                        sub_list = split_list[i]
                        tmp_label_class = eval('new_label_class_{}'.format(i))
                        if set(label_class).issubset(set(sub_list)):
                            if c in sub_list:
                                tmp_label_class.append(c)
        else:
            for c in label_class:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0],target_pix[1]] = 1 
                
                if tmp_label.sum() >= 2 * 32 * 32:   
                    all_label_class.append(c)
                    for i in range(len(split_list)):
                        sub_list = split_list[i]
                        tmp_label_class = eval('new_label_class_{}'.format(i))
                        if c in sub_list:
                            tmp_label_class.append(c)    
   
        for i in range(len(split_list)):
            new_label_class = eval('new_label_class_{}'.format(i))
            image_label_list = eval('image_label_list_{}'.format(i))
            if len(new_label_class) > 0:
                image_label_list.append(item)

        for c in all_label_class:
            sub_class_file_list[c].append(item)
            
    for i in range(len(split_list)):
        image_label_list = eval('image_label_list_{}'.format(i))
        fin_list.append(image_label_list)  
    # print("Checking image&label pair {} list done! ".format(split))
    return fin_list, sub_class_file_list

def make_data_list(data_root=None, dataset=None, train_list=None, val_list=None, all_class=None, val_class=None, ):
    list_root = './lists/{}/fss_list/'.format(dataset)
    if not os.path.exists(list_root):
        train_root = list_root + 'train/'
        val_root = list_root + 'val/'
        print('{} has been created!'.format(list_root))
        os.makedirs(train_root)
        os.makedirs(val_root)

    data_root = data_root

    train_class_list = []
    for i in range(len(val_class)):
        tmp_list = list(set(all_class) - set(val_class[i]))
        train_class_list.append(tmp_list)


    print('Processing train_split')
    train_txt, train_dict = make_dataset(data_root=data_root, data_list=train_list, all_class=all_class, split_list=train_class_list, fliter_intersection=True)
    with open (train_root + 'train_dict.txt', 'w') as f:
        f.write(str(train_dict))

    for i in range(len(train_txt)):
        data_list = train_txt[i]
        with open(train_root + 'train_split{}.txt'.format(i), 'w')as f:
            for item in data_list:
                img, label = item
                f.write(img + ' ')
                f.write(label + '\n')
            
    print('Processing val_split')
    val_txt, val_dict = make_dataset(data_root=data_root, data_list=val_list, all_class=all_class, split_list=val_class, fliter_intersection=False)
    with open (val_root + 'val_dict.txt', 'w') as f:
        f.write(str(val_dict))

    for i in range(len(val_txt)):
        data_list = val_txt[i]
        with open(val_root + 'val_split{}.txt'.format(i), 'w')as f:
            for item in data_list:
                img, label = item
                f.write(img + ' ')
                f.write(label + '\n')

    print('Processing val_base')
    base_txt, _ = make_dataset(data_root=data_root, data_list=val_list, all_class=all_class, split_list=val_class, fliter_intersection=False)

    for i in range(len(base_txt)):
        data_list = base_txt[i]
        with open(val_root + 'val_base{}.txt'.format(i), 'w')as f:
            for item in data_list:
                img, label = item
                f.write(img + ' ')
                f.write(label + '\n')

def make_dict(data_root=None, data_list=None, all_class=None, dataset=None, mode=None):    

    """
    data_list: all data in train/val
    all_class: all class in dataset
    
    return the dict that contains the data_list of each classes
    """

    dict_name = './lists/{}/{}_dict.txt'.format(dataset, mode)
    list_read = open(data_list).readlines()

    sub_class_file_list = {}
    for sub_c in all_class:
        sub_class_file_list[sub_c] = []
    print('Processing {} data' .format(mode))
    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)
        label = cv2.imread(label_name, cv2.IMREAD_GRAYSCALE)
        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        all_label_class = []
        for c in label_class:
            tmp_label = np.zeros_like(label)
            target_pix = np.where(label == c)
            tmp_label[target_pix[0],target_pix[1]] = 1 
            
            if tmp_label.sum() >= 2 * 32 * 32:   
                all_label_class.append(c)

        for c in all_label_class:
            sub_class_file_list[c].append(item)

    with open (dict_name, 'w') as f:
        f.write(str(sub_class_file_list))

def gen_list(class_dict_name, class_list, fliter):
    with open(class_dict_name, 'r') as f:
        f_str = f.read()
    class_dict = eval(f_str)

    Discard = set()
    Adopt = set()
    for id in class_dict:
        if id in class_list:
            Adopt = set(class_dict[id]) | Adopt
        else:
            Discard = set(class_dict[id]) | Discard
    if fliter:
        out_list = Adopt - Discard
    else:
        out_list = Adopt
    
    return list(out_list), class_dict

    # return sub_class_file_list

