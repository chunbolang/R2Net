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
from util.util import AverageMeter,  intersectionAndUnionGPU, get_model_para_number, setup_seed, get_logger, get_save_path, \
                                     fix_bn, check_makedirs,freeze_modules,lr_decay, Special_characters

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Few-Shot Semantic Segmentation')
    parser.add_argument('--arch', type=str, default='R2Net', help='') # 
    parser.add_argument('--shot', type=int, default=1, help='') # 
    parser.add_argument('--split', type=int, default=0, help='') # 
    parser.add_argument('--dataset', type=str, default='iSAID', help='') # 
    parser.add_argument('--backbone', type=str, default='vgg', help='') # 
    parser.add_argument('--variable1', type=str, default='', help='') #
    parser.add_argument('--variable2', type=str, default='', help='') #
    parser.add_argument('--local_rank', type=int, default=-1, help='number of cpu threads to use during batch generation')    
    parser.add_argument('--opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    base_config = 'config/base.yaml'
    data_config = 'config/dataset/{}.yaml'.format(args.dataset)
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

    freeze_modules(model, args.freeze_layer)
    time.sleep(2)

    if args.distributed:
        # Initialize Process Group
        device = torch.device('cuda', args.local_rank)
        model.to(device)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model = model.cuda()

    # Resume
    if args.resume:
        resume_path = osp.join(args.snapshot_path, args.resume)
        if os.path.isfile(resume_path):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location=torch.device('cpu'))
            args.start_epoch = checkpoint['epoch']
            new_param = checkpoint['state_dict']
            try: 
                model.load_state_dict(new_param)
            except RuntimeError:                   # 1GPU loads mGPU model
                for key in list(new_param.keys()):
                    new_param[key[7:]] = new_param.pop(key)
                model.load_state_dict(new_param)
            optimizer.load_state_dict(checkpoint['optimizer'])
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))
        else:
            if main_process():       
                logger.info("=> no checkpoint found at '{}'".format(resume_path))


    # Get model para.
    total_number, learnable_number = get_model_para_number(model)
    if main_process():
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
    args.distributed = False # Debug
    # args.distributed = True if torch.cuda.device_count() > 1 else False
    
    shuffle = False if args.distributed else True
    

    get_save_path(args)
    check_makedirs(args.snapshot_path)
    check_makedirs(args.result_path)

    if main_process():
        print(args)

    if args.manual_seed is not None:
        setup_seed(args.manual_seed, args.seed_deterministic)
        
    if args.distributed:
        dist.init_process_group(backend='nccl')
        print('args.local_rank: ', args.local_rank)
        torch.cuda.set_device(args.local_rank)

    if main_process():
        logger.info("=> creating dataset ...")

# ----------------------  DATASET  ----------------------
    train_data = eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, \
                                            shot=args.shot, mode='train', transform_dict=args.train_transform)
    train_sampler = DistributedSampler(train_data) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=shuffle, \
                                            num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    # Val

    val_data =  eval('{}.{}_few_dataset'.format(args.dataset, args.dataset))(split=args.split, \
                                            shot=args.shot, mode='val', transform_dict=args.val_transform)            
    val_sampler = DistributedSampler(val_data) if args.distributed else None                  
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False, \
                                            num_workers=args.workers, pin_memory=False, sampler=val_sampler)
    logger.info('train_list: {}'.format(train_data.list))
    logger.info('num_train_data: {}'.format(len(train_data)))
    logger.info('val_list: {}'.format(val_data.list))
    logger.info('num_val_data: {}'.format(len(val_data)))

    args.base_class_num = len(train_data.list)
    args.novel_class_num = len(val_data.list)

    config_file = args.snapshot_path + 'config.yaml'

    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(str(args))
    time.sleep(2)

    if main_process():
        logger.info("=> creating model ...")
    model, optimizer = get_model(args)
    
    logger.info(model)


# ----------------------  TRAINVAL  ----------------------
    global best_miou, best_FBiou, best_epoch, keep_epoch, val_num
    global best_name, grow_name, all_name, latest_name, best_class_iou

    best_miou = 0.
    best_FBiou = 0.
    best_class_iou = []
    best_epoch = 0
    keep_epoch = 0
    val_num = 0

    start_time = time.time()
    scaler = amp.GradScaler()

#--------------------------- FilenamePrepare -----------------------------

    latest_name = args.snapshot_path + 'latest.pth'
    best_name = args.snapshot_path + 'best.pth'
    grow_name = args.snapshot_path + 'grow.txt'
    all_name = args.snapshot_path + 'all.txt'

    for epoch in range(args.start_epoch, args.epochs):
        if keep_epoch == args.stop_interval:
            break
        if args.fix_random_seed_val:
            setup_seed(args.manual_seed + epoch, args.seed_deterministic)

        epoch_log = epoch + 1
        keep_epoch += 1
        
        # ----------------------  TRAIN  ----------------------
        train(train_loader, val_loader, model, optimizer, epoch, scaler)
        torch.cuda.empty_cache()
        # save model for <resuming>
        if ((epoch + 1) % args.save_freq == 0) and main_process():

            logger.info('Saving checkpoint to: ' + latest_name)
            if osp.exists(latest_name):
                os.remove(latest_name)            
            torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, latest_name)


        # -----------------------  VAL  -----------------------
        if args.evaluate and (epoch + 1)% args.val_freq == 0:
            _,fbIou, _,_, mIoU,_ , recall, precision, class_miou = validate(val_loader, model)   
            torch.cuda.empty_cache()
            val_num += 1

            with open(all_name, 'a') as f:
                f.write('[{},miou:{:.4f}, fbIou:{:.4f}, recall:{:.4f}, precision:{:.4f},]\n'.format(epoch+1, mIoU, fbIou, recall, precision))

        # save model for <testing> and <fine-tuning>
            if mIoU > best_miou:
                best_miou, best_epoch, best_class_iou, best_FBiou = mIoU, epoch, class_miou, fbIou
                keep_epoch = 0
                with open(grow_name, 'a') as f:
                    f.write('Best_epoch:{} , Best_miou:{:.4f} , fbIou:{:.4f} , recall:{:.4f}, precision:{:.4f}, \n'.format(epoch+1 , best_miou, fbIou, recall, precision)) 
                logger.info('Saving checkpoint to: ' + best_name + '  miou: {:.4f}'.format(best_miou))
                if osp.exists(best_name):
                    os.remove(best_name)    
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_name)  

    with open(args.snapshot_path + 'class.txt', 'a') as f:
        for i in range(len(best_class_iou)):
            f.write('{:.4f}\t'.format(best_class_iou[i]))
        f.write('\nmiou: {:.4f}, fb_iou: {:.4f}'.format(best_miou, best_FBiou))
    total_time = time.time() - start_time
    t_m, t_s = divmod(total_time, 60)
    t_h, t_m = divmod(t_m, 60)
    total_time = '{:02d}h {:02d}m {:02d}s'.format(int(t_h), int(t_m), int(t_s))

    print('\nEpoch: {}/{} \t Total running time: {}'.format(epoch_log, args.epochs, total_time))
    print('The number of models validated: {}'.format(val_num))            
    print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  Final Best Result   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    print(args.arch + '\t Group:{} \t Best_mIoU:{:.4f} \t Best_FBIoU:{:.4f} \t Best_step:{}'.format(args.split, best_miou, best_FBiou, best_epoch + 1 ))
    print('>'*80)
    print ('当前的日期和时间是 %s' % datetime.datetime.now())


def train(train_loader, val_loader, model, optimizer, epoch ,scaler):
    global best_miou, best_epoch, keep_epoch, val_num
    batch_time = AverageMeter()
    data_time = AverageMeter()
    main_loss_meter = AverageMeter()
    aux_loss_meter_1 = AverageMeter()
    aux_loss_meter_2 = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    recall_meter = AverageMeter()
    acc_meter = AverageMeter()
    tmp_num = 0
    model.train()
    if args.fix_bn:
        model.apply(fix_bn) # fix batchnorm

    end = time.time()
    val_time = 0.
    max_iter = args.epochs * len(train_loader)
    
    current_characters = Special_characters[random.randint(0,len(Special_characters)-1)]

    current_GPU = os.environ["CUDA_VISIBLE_DEVICES"]
    # GPU_name = torch.cuda.get_device_name()
    for i, (input, target, s_input, s_mask, subcls) in enumerate(train_loader):

        data_time.update(time.time() - end - val_time)
        current_iter = epoch * len(train_loader) + i + 1

        lr_decay(optimizer, args.base_lr, current_iter, max_iter, args.lr_decay, current_characters )
        if current_iter % 50 == 0 and main_process():
            print(current_characters[0]*3 +' '*5 + '{}_{}_{}_split{}_{}shot Pretrain: {} GPU_id: {}'.format(args.arch,\
                             args.dataset ,args.backbone, args.split, args.shot, args.pretrain, current_GPU) + ' '*5 + current_characters[1]*3)
   
        s_input = s_input.cuda(non_blocking=True)
        s_mask = s_mask.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad()
        # with torch.autograd.set_detect_anomaly(True):      # debug
        output, main_loss, aux_loss_1, aux_loss_2= model(s_x=s_input, s_y=s_mask, x=input, y=target, cat_idx=subcls)
        loss = main_loss + aux_loss_1 * args.aux_weight1 +aux_loss_2 * args.aux_weight2
    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if 'para_limit' in args.keys():
            for item_id in range(len(args.para_limit.name)):
                item = args.para_limit.name[item_id]
                tmp_limit = args.para_limit.limit[item_id]
                eval('model.{}'.format(item)).data.clamp_(tmp_limit[0], tmp_limit[1])

        n = input.size(0) # batch_size

        intersection, union, target = intersectionAndUnionGPU(output, target, 2, args.ignore_label)
        intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)
        
        
        accuracy = sum(intersection_meter.val[1:]) / (sum(target_meter.val[1:]) + 1e-10)  # allAcc
        main_loss_meter.update(main_loss.item(), n)
        if isinstance(aux_loss_1, torch.Tensor):
            aux_loss_meter_1.update(aux_loss_1.item(), n)
        if isinstance(aux_loss_2, torch.Tensor):
            aux_loss_meter_2.update(aux_loss_2.item(), n)

        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end - val_time)
        end = time.time()

        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'MainLoss {main_loss_meter.val:.4f} '
                        'AuxLoss_1 {aux_loss_meter_1.val:.4f} '  
                        'AuxLoss_2 {aux_loss_meter_2.val:.4f} ' 
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch+1, args.epochs, i + 1, len(train_loader),
                                                        batch_time=batch_time,
                                                        data_time=data_time,
                                                        remain_time=remain_time,
                                                        main_loss_meter=main_loss_meter,
                                                        aux_loss_meter_1=aux_loss_meter_1,
                                                        aux_loss_meter_2=aux_loss_meter_2,
                                                        loss_meter=loss_meter,
                                                        accuracy=accuracy))


        
        # -----------------------  SubEpoch VAL  -----------------------
        if args.evaluate and args.SubEpoch_val and ((epoch + 1)%args.val_freq==0) and (i in torch.arange(1,args.sub_freq)*round(len(train_loader)/args.sub_freq)): # max_epoch<=100时进行half_epoch Val
            _,fbIou, _,_, mIoU,_ , recall, precision, class_miou = validate(val_loader, model)    
            torch.cuda.empty_cache()
            model.train()
            if args.fix_bn:
                model.apply(fix_bn)
            val_num += 1 
            tmp_num += 1
            # save model for <testing> and <fine-tuning>

            with open(all_name, 'a') as f:
                f.write('[{}_{},miou:{:.4f}, fbIou:{:.4f}, recall:{:.4f}, precision:{:.4f},]\n'.format(epoch, tmp_num, mIoU, fbIou, recall, precision))

            if mIoU > best_miou:
                best_miou, best_epoch, best_class_iou, best_FBiou = mIoU, epoch+(1/args.sub_freq)*tmp_num, class_miou, fbIou
                keep_epoch = 0
                with open(grow_name, 'a') as f:
                    f.write('Best_epoch:{}_{} , Best_miou:{:.4f} , fbIou:{:.4f} , recall:{:.4f}, precision:{:.4f}, \n'.format(epoch, tmp_num, best_miou, fbIou, recall, precision)) 
             
                logger.info('Saving checkpoint to: ' + best_name + '  miou: {:.4f}'.format(best_miou))
                if osp.exists(best_name):
                    os.remove(best_name) 
                torch.save({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, best_name) 
            
    if hasattr(model, 'out_data'):
        with open(args.snapshot_path + 'out_data.txt', 'a') as f:
            for item in model.out_data:
                f.write(item + '\n')
            model.out_data = []

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(epoch+1, args.epochs, mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))

    return main_loss_meter.avg, mIoU, mAcc, allAcc

def validate(val_loader, model):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    model_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    split_gap = len(val_loader.dataset.list)
    test_num = 1000 # 20000 

    class_intersection_meter = [0]*split_gap
    class_union_meter = [0]*split_gap   
    class_target_meter = [0]*split_gap   

    if args.manual_seed is not None and args.fix_random_seed_val:
        setup_seed(args.manual_seed, args.seed_deterministic)

    criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)

    model.eval()
    end = time.time()
    val_start = end

    assert test_num % args.batch_size_val == 0
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

            # for b_id in range(output.size(0)):
            intersection, union, new_target = intersectionAndUnionGPU(output, target, 2, args.ignore_label)
            intersection, union, new_target = intersection.cpu().numpy(), union.cpu().numpy(), new_target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(new_target)

            tmp_id = subcls[0].cpu().numpy()[0]
            class_intersection_meter[tmp_id] += intersection[1]
            class_union_meter[tmp_id] += union[1] 
            class_target_meter[tmp_id] += new_target[1]

            recall = np.mean(intersection_meter.val[1:] / (target_meter.val[1:]+ 1e-10))
            precision = np.mean(intersection_meter.val[1:] /(union_meter.val[1:] - target_meter.val[1:] + intersection_meter.val[1:] + 1e-10) ) 
            Iou = sum(intersection_meter.val[1:]) / (sum(union_meter.val[1:]) + 1e-10)

            loss_meter.update(loss.item(), input.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if ((iter_num) % round((alpha/20)) == 0):
                logger.info('Test: [{}/{}] '
                            'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                            'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                            'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                            'recall {recall:.4f} '
                            'precision {precision:.4f} '
                            'Iou {Iou:.4f}.'.format(iter_num* args.batch_size_val, test_num,
                                                            data_time=data_time,
                                                            batch_time=batch_time,
                                                            loss_meter=loss_meter,
                                                            recall=recall,
                                                            precision=precision,
                                                            Iou=Iou))
    val_time = time.time()-val_start

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    
    class_iou_class = []
    class_miou = 0
    class_recall_class = []
    class_mrecall = 0
    class_precisoin_class = []
    class_mprecision = 0

    for i in range(len(class_intersection_meter)):
        class_iou = class_intersection_meter[i]/(class_union_meter[i]+ 1e-10)
        class_iou_class.append(class_iou)
        class_miou += class_iou
        
        class_recall = class_intersection_meter[i]/(class_target_meter[i]+ 1e-10)
        class_recall_class.append(class_recall)
        class_mrecall += class_recall

        class_precision = class_intersection_meter[i]/(class_union_meter[i] - class_target_meter[i] + class_intersection_meter[i]+ 1e-10)
        class_precisoin_class.append(class_precision)
        class_mprecision += class_precision

    class_mrecall = class_mrecall*1.0 / len(class_intersection_meter)  
    class_miou = class_miou*1.0 / len(class_intersection_meter)
    class_mprecision = class_mprecision*1.0 / len(class_intersection_meter)


    logger.info('mean IoU---Val result: mIoU {:.4f}.'.format(class_miou))
    logger.info('mean recall---Val result: mrecall {:.4f}.'.format(class_mrecall))
    logger.info('mean precisoin---Val result: mprecisoin {:.4f}.'.format(class_mprecision))

    for i in range(split_gap):
        logger.info('Class_{}: \t Result: iou {:.4f}. \t recall {:.4f}. \t precision {:.4f}. \t {}'.format(i+1, \
                        class_iou_class[i], class_recall_class[i], class_precisoin_class[i],\
                         val_loader.dataset.class_id[val_loader.dataset.list[i]]))     

     

    logger.info('FBIoU---Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(2):
        logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    print('total time: {:.4f}, avg inference time: {:.4f}, count: {}'.format(val_time, model_time.avg, test_num))

    return loss_meter.avg, mIoU, mAcc, allAcc, class_miou, iou_class[1], class_mrecall, class_mprecision, class_iou_class


if __name__ == '__main__':
    main()
