import numpy as np
import cv2
import os
import time
from tqdm import tqdm
# from dataset.base import classId2className
classId2className = {
                    'coco': {
                         0: 'background',
                         1: 'person',
                         2: 'bicycle',
                         3: 'car',
                         4: 'motorcycle',
                         5: 'airplane',
                         6: 'bus',
                         7: 'train',
                         8: 'truck',
                         9: 'boat',
                         10: 'traffic light',
                         11: 'fire hydrant',
                         12: 'stop sign',
                         13: 'parking meter',
                         14: 'bench',
                         15: 'bird',
                         16: 'cat',
                         17: 'dog',
                         18: 'horse',
                         19: 'sheep',
                         20: 'cow',
                         21: 'elephant',
                         22: 'bear',
                         23: 'zebra',
                         24: 'giraffe',
                         25: 'backpack',
                         26: 'umbrella',
                         27: 'handbag',
                         28: 'tie',
                         29: 'suitcase',
                         30: 'frisbee',
                         31: 'skis',
                         32: 'snowboard',
                         33: 'sports ball',
                         34: 'kite',
                         35: 'baseball bat',
                         36: 'baseball glove',
                         37: 'skateboard',
                         38: 'surfboard',
                         39: 'tennis racket',
                         40: 'bottle',
                         41: 'wine glass',
                         42: 'cup',
                         43: 'fork',
                         44: 'knife',
                         45: 'spoon',
                         46: 'bowl',
                         47: 'banana',
                         48: 'apple',
                         49: 'sandwich',
                         50: 'orange',
                         51: 'broccoli',
                         52: 'carrot',
                         53: 'hot dog',
                         54: 'pizza',
                         55: 'donut',
                         56: 'cake',
                         57: 'chair',
                         58: 'sofa',
                         59: 'pottedplant',
                         60: 'bed',
                         61: 'diningtable',
                         62: 'toilet',
                         63: 'tv',
                         64: 'laptop',
                         65: 'mouse',
                         66: 'remote',
                         67: 'keyboard',
                         68: 'cell phone',
                         69: 'microwave',
                         70: 'oven',
                         71: 'toaster',
                         72: 'sink',
                         73: 'refrigerator',
                         74: 'book',
                         75: 'clock',
                         76: 'vase',
                         77: 'scissors',
                         78: 'teddy bear',
                         79: 'hair drier',
                         80: 'toothbrush'},

                    'pascal': {
                        0: 'background',
                        1: 'airplane',
                        2: 'bicycle',
                        3: 'bird',
                        4: 'boat',
                        5: 'bottle',
                        6: 'bus',
                        7: 'cat',
                        8: 'car',
                        9: 'chair',
                        10: 'cow',
                        11: 'diningtable',
                        12: 'dog',
                        13: 'horse',
                        14: 'motorcycle',
                        15: 'person',
                        16: 'pottedplant',
                        17: 'sheep',
                        18: 'sofa',
                        19: 'train',
                        20: 'tv'
                        },
                    
                    'iSAID':{
                        0: 'unlabeled',
                        1: 'ship',
                        2: 'storage_tank',
                        3: 'baseball_diamond',
                        4: 'tennis_court',
                        5: 'basketball_court',
                        6: 'Ground_Track_Field',
                        7: 'Bridge',
                        8: 'Large_Vehicle',
                        9: 'Small_Vehicle',
                        10: 'Helicopter',
                        11: 'Swimming_pool',
                        12: 'Roundabout',
                        13: 'Soccer_ball_field',
                        14: 'plane',
                        15: 'Harbor'
                    },

                    'few_shot':{
                        0: 'Background',
                        1: 'Foreground',

                    }
                     }


dataset = 'iSAID'
data_root = '/disk2/lcb/datasets/{}/ann_dir/'.format(dataset)

list_dir = 'lists/{}/'.format(dataset)
if not os.path.exists(list_dir):
    os.makedirs(list_dir)

class_num = len(classId2className[dataset]) -1

def gen_list(dataset, data_root, list_dir, class_num):

    for mode in ['train', 'val']:
        mutli_class_num = np.zeros(class_num)
        pot_num = np.zeros(class_num)
        pic_num= np.zeros(class_num)

        tmp_root = data_root + mode
        new_root = tmp_root #.replace('ann_dir', 'ann_dir_1')

        file_list = os.listdir(new_root)
        pair_list = []
        for idx in tqdm(range(len(file_list))):
            file = file_list[idx]
            label_name = 'ann_dir/{}/'.format(mode) +  file
            image_name = ('img_dir/{}/'.format(mode) + file.replace('_instance_color_RGB', ''))
            label = cv2.imread(new_root + '/' + file, cv2.IMREAD_GRAYSCALE)
            label_list = np.unique(label).tolist()
            if 0 in label_list:
                label_list.remove(0)
            if 255 in label_list:
                label_list.remove(255)

            if len(label_list) != 0 :  #and len(label_list) <3
                mutli_class_num[len(label_list)-1] += 1

                for cls in label_list:
                    if len(np.where(label == cls)[0]) > 2*32*32 :
                        pair_list.append(image_name + ' ' + label_name)
                        
                        pic_num[cls-1] += 1
                        pot_num[cls-1] += len(np.where(label == cls)[0])/1000000

        pair_list = list(set(pair_list))

        with open(list_dir + '{}_num.txt'.format(mode), 'a') as f:
            f.write('-'*23 +'{}_{}'.format( dataset, mode) + '-'*23  + '\n')
            for i in range(class_num):
                f.write('同时含有{}个类别的图像数目：{}'.format(i+1, mutli_class_num[i]) + '\n')
            f.write('图像总数：{}'.format(np.sum(mutli_class_num) )+ '\n')

            for i in range(class_num):
                f.write('类别{} \t 图像数目：{} \t {} '.format(i+1,  pic_num[i], classId2className[dataset][i+1]) + '\n')

            f.write('图像数目类别比列为：{}'. format(np.round(pic_num/np.min(pic_num), 2))  + '\n')

            for i in range(class_num):
                f.write('类别{} \t 像素点数目：{:.1f} million \t {}'.format(i+1, pot_num[i], classId2className[dataset][i+1]) + '\n')

            f.write('像素点类别比列为：{}'. format(np.round(pot_num/np.min(pot_num), 2))  + '\n')
        
        with open(list_dir + '{}.txt'.format(mode), 'a') as f:
            for pair in pair_list:
                f.write(pair + '\n')
                
def change_label(data_root):
    for mode in ['train', 'val']:
        tmp_root = data_root + mode
        new_root = tmp_root.replace('ann_dir', 'ann_dir_1')
        if not os.path.exists(new_root):
            os.makedirs(new_root)
        file_list = os.listdir(tmp_root)
        for idx in tqdm(range(len(file_list))):
            file = file_list[idx]
            label_name = new_root + '/' + file
            label = cv2.imread(tmp_root + '/' + file, cv2.IMREAD_GRAYSCALE)
            label[label==0] = 255
            label[label==6] = 0
            cv2.imwrite(label_name, label)


# change_label(data_root)
gen_list(dataset, data_root, list_dir, class_num)
