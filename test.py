import os
import datetime
import random
import time
import cv2
import numpy as np
import logging
import argparse
from visdom import Visdom
import os.path as osp

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.cuda.amp import autocast as autocast
from torch.cuda import amp
from torch.utils.data.distributed import DistributedSampler

from model.few_seg import R2Net
# from model.workdir import
from dataset import iSAID, iSAID_1

from util import config
from util.util import AverageMeter, poly_learning_rate, intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                    is_same_model, fix_bn, sum_list, check_makedirs,freeze_modules, adjust_learning_rate_poly,lr_decay

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='R2Net', help='') # 
    parser.add_argument('--shot', type=int, default=1, help='') # 
    parser.add_argument('--split', type=int, default=0, help='') # 
    parser.add_argument('--dataset', type=str, default='iSAID', help='') # 
    parser.add_argument('--backbone', type=str, default='vgg', help='') # 
    parser.add_argument('--s_q', default='False', help='') #
    parser.add_argument('--cross_domain', default=None, help='') #
    parser.add_argument('--variable1', type=str, default='', help='') #
    parser.add_argument('--variable2', type=str, default='', help='') #
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base_config = 'config/base.yaml'
    data_config = 'config/dataset/{}.yaml'.format(args.cross_domain)
    if args.arch in ['R2Net']:
        model_config = 'config/model/few_seg/{}.yaml'.format(args.arch)
    else:
        model_config = 'config/model/workdir/{}.yaml'.format(args.arch)

    if os.path.exists(model_config):
        cfg = config.load_cfg_from_cfg_file([base_config, data_config, model_config])
    else:
        cfg = config.load_cfg_from_cfg_file([base_config, data_config])

    cfg = config.merge_cfg_from_args(cfg, args)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_model(args):

    model = eval(args.arch).OneModel(args, cls_type='Base')
    optimizer = model.get_optim(model, args.lr_decay, LR=args.base_lr)

    model = model.cuda()

    # Resume
    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    weight_path = osp.join(args.snapshot_path, 'best.pth')
    if os.path.isfile(weight_path):
        logger.info("=> loading checkpoint '{}'".format(weight_path))
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        new_param = checkpoint['state_dict']
        try: 
            model.load_state_dict(new_param)
        except RuntimeError:                   # 1GPU loads mGPU model
            for key in list(new_param.keys()):
                new_param[key[7:]] = new_param.pop(key)
            model.load_state_dict(new_param)
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(weight_path, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(weight_path))

    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    print('Number of Parameters: %d' % (total_number))
    print('Number of Learnable Parameters: %d' % (learnable_number))

    time.sleep(5)
    return model, optimizer

def main_process():
    return not args.distributed or (args.distributed and (args.local_rank == 0))

def main():
    global args, logger
    args = get_parser()
    logger = get_logger()
    args.logger = logger
    args.distributed = True if torch.cuda.device_count() > 1 else False

    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)

    if main_process():
        logger.info("=> creating dataset ...")
        
    train_data = eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, \
                                        shot=args.shot, mode='train', transform_dict=args.train_transform)
    train_id = []
    for i in range(len(train_data.list)):
        train_id.append(train_data.class_id[train_data.list[i]])
    val_data =  eval('{}.{}_few_dataset'.format(args.cross_domain, args.cross_domain))(split=args.split, \
                                        shot=args.shot, mode='val', transform_dict=args.val_transform)
    val_data.list = []
    for id in val_data.all_class:
        if val_data.class_id[id] not in train_id:
            val_data.list.append(id)
    tmp = set() 
    for item in val_data.list:
        tmp = tmp | set(val_data.sub_class_file_list[item])
    val_data.data_list = list(tmp)
        
    val_sampler = DistributedSampler(val_data) if args.distributed else None                  
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, \
                                            num_workers=args.workers, pin_memory=False, sampler=val_sampler)

    args.base_class_num =len(train_data.list)
    args.novel_class_num = len(val_data.list)

    # if args.cross_dataset:
    #     train_data = eval('{}.{}_few_dataset'.format(args.train_dataset, args.train_dataset)).class_id

    logger.info('val_list: {}'.format(val_data.list))
    logger.info('num_val_data: {}'.format(len(val_data)))

    if main_process():
        logger.info("=> creating model ...")
    model, _ = get_model(args)
    
    logger.info(model)

    val_manual_seed = args.manual_seed
    if eval(args.s_q):
        val_num = 2
    else:
        val_num = 5

    setup_seed(val_manual_seed, False)
    seed_array = np.random.randint(0,1000,val_num) 

    start_time = time.time()
    FBIoU_array = np.zeros(val_num)
    mIoU_array = np.zeros(val_num)
    pIoU_array = np.zeros(val_num)
    class_array = np.zeros([val_num, len(val_data.list)])

    txt_root = 'exp/{}/test_result/'.format(args.arch)
    check_makedirs(txt_root)

    for val_id in range(val_num):
        val_seed = seed_array[val_id]
        print('Val: [{}/{}] \t Seed: {}'.format(val_id+1, val_num, val_seed))
        loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou, pIoU, class_iou = validate(val_loader, model, val_seed) 

        FBIoU_array[val_id], mIoU_array[val_id], pIoU_array[val_id] = mIoU_val, class_miou, pIoU
        for class_id in range(len(class_iou)):
            class_array[val_id, class_id] = class_iou[class_id]
    class_marray = np.mean(class_array, 0)
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nTotal running time: {}'.format(total_time))
    print('Seed0: {}'.format(val_manual_seed))
    print('mIoU:  {}'.format(np.round(mIoU_array, 4)))
    print('FBIoU: {}'.format(np.round(FBIoU_array, 4)))
    print('pIoU:  {}'.format(np.round(pIoU_array, 4)))
    print('-'*43)
    print('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {}'.format(seed_array[mIoU_array.argmax()], seed_array[FBIoU_array.argmax()], seed_array[pIoU_array.argmax()]))
    print('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f}'.format(mIoU_array.max(), FBIoU_array.max(), pIoU_array.max()))
    print('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_pIoU: {:.4f}'.format(mIoU_array.mean(), FBIoU_array.mean(), pIoU_array.mean()))

    with open(txt_root + '{}_{}.txt'.format(args.dataset, args.cross_domain), 'a') as f:
        f.write('\nsupp=query : {}  '.format(args.s_q)+ '\n')
        f.write('{} {} split{} {} shot'.format(args.arch, args.backbone, args.split, args.shot) + '\n')
        f.write('Seed0: {}\n'.format(val_manual_seed))
        f.write('Seed:  {}\n'.format(seed_array))
        f.write('mIoU:  {}\n'.format(np.round(mIoU_array, 4)))
        f.write('FBIoU: {}\n'.format(np.round(FBIoU_array, 4)))
        f.write('pIoU:  {}\n'.format(np.round(pIoU_array, 4)))
        f.write('Best_Seed_m: {} \t Best_Seed_F: {} \t Best_Seed_p: {} \n'.format(seed_array[mIoU_array.argmax()], seed_array[FBIoU_array.argmax()], seed_array[pIoU_array.argmax()]))
        f.write('Best_mIoU: {:.4f} \t Best_FBIoU: {:.4f} \t Best_pIoU: {:.4f} \n'.format(mIoU_array.max(), FBIoU_array.max(), pIoU_array.max()))
        f.write('Mean_mIoU: {:.4f} \t Mean_FBIoU: {:.4f} \t Mean_pIoU: {:.4f} \n'.format(mIoU_array.mean(), FBIoU_array.mean(), pIoU_array.mean()))
        for id in range(len(val_data.list)):
            f.write('{}\t'.format(val_data.list[id]))
        f.write('\n')
        for id in range(len(val_data.list)):
            f.write('{:.2f}\t'.format(class_marray[id]*100))
        f.write('\n' + '-'*47 + '\n')
        f.write(str(datetime.datetime.now()) + '\n')

def validate(val_loader, model, val_seed):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = len(val_loader.dataset.list)
    if args.s_q:
        test_num = min(len(val_loader)*2, 1000)
    else:
        test_num = 1000

    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap   

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    setup_seed(val_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    alpha = round(test_num / args.batch_size_val)
    iter_num = 0
    total_time = 0
    for e in range(10):
        for i, (input, target, s_input, s_mask, subcls, ori_label) in enumerate(val_loader):
            if iter_num * args.batch_size_val >= test_num:
                break
            iter_num += 1
            data_time.update(time.time() - end)

            s_input = s_input.cuda(non_blocking=True)
            s_mask = s_mask.cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            ori_label = ori_label.cuda(non_blocking=True)
            start_time = time.time()

            # with autocast():
            if eval(args.s_q):
                assert (args.shot==1)
                with torch.no_grad():
                    output = model(s_x=input.unsqueeze(1), s_y=target.unsqueeze(1), x=input, y=target, cat_idx=subcls)
            else:
                with torch.no_grad():
                    output = model(s_x=s_input, s_y=s_mask, x=input, y=target, cat_idx=subcls)
            total_time = total_time + 1
            model_time.update(time.time() - start_time)

            if args.ori_resize:
                longerside = max(ori_label.size(1), ori_label.size(2))
                backmask = torch.ones(ori_label.size(0), longerside, longerside, device='cuda')*255
                backmask[:, :ori_label.size(1), :ori_label.size(2)] = ori_label
                target = backmask.clone().long()

            output = F.interpolate(output, size=target.size()[1:], mode='bilinear', align_corners=True)
            output = output.float()
            loss = criterion(output, target)    

            n = input.size(0)
            loss = torch.mean(loss)

            output = output.max(1)[1]
            for b_id in range(output.size(0)):
                intersection, union, new_target = intersectionAndUnionGPU(output[b_id,], target[b_id,], 2, args.ignore_label)
                intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
                intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

                tmp_id = subcls[0].cpu().numpy()[b_id]
                class_intersection_meter[tmp_id] += intersection[1]
                class_union_meter[tmp_id] += union[1] 

            accuracy = sum(intersection_meter.val[1:]) / (sum(target_meter.val[1:]) + 1e-10)

            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((iter_num ) % round((alpha/20)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'Accuracy {accuracy:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            accuracy=accuracy))
    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    
    class_iou_class = []
    class_miou = 0
    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    logger.info('meanIoU---Val result: mIoU {:.4f}.'.format(class_miou))
    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t {}'.format(i+1, class_iou_class[i],\
                         val_loader.dataset.class_id[val_loader.dataset.list[i]]))            
     

    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, iou_class[1], class_iou_class


if __name__ == '__main__':
    main()
