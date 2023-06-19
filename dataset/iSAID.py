from dataset.base_data import Few_Data, Base_Data


class iSAID_few_dataset(Few_Data):

    class_id = {
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
                    }
    
    PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]
    
    all_class = list(range(1, 16))
    val_class = [list(range(1, 6)), list(range(6, 11)), list(range(11, 16))]

    data_root = '../data/iSAID'
    train_list ='./lists/iSAID/train.txt'
    val_list ='./lists/iSAID/val.txt'

    def __init__(self, split=0, shot=1, dataset='iSAID', mode='train', ann_type='mask', transform_dict=None):
        super().__init__(split, shot, dataset, mode, ann_type, transform_dict)


class iSAID_base_dataset(Base_Data):
    class_id = {
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
                    }
    
    PALETTE = [[0, 0, 0], [0, 0, 63], [0, 63, 63], [0, 63, 0], [0, 63, 127],
               [0, 63, 191], [0, 63, 255], [0, 127, 63], [0, 127, 127],
               [0, 0, 127], [0, 0, 191], [0, 0, 255], [0, 191, 127],
               [0, 127, 191], [0, 127, 255], [0, 100, 155]]
    
    all_class = list(range(1, 16))
    val_class = [list(range(1, 6)), list(range(6, 11)), list(range(11, 16))]

    data_root = '../data/iSAID'
    train_list ='./lists/iSAID/train.txt'
    val_list ='./lists/iSAID/val.txt'

    def __init__(self, split=0, shot=1, data_root=None, dataset='iSAID', mode='train', transform_dict=None):
        super().__init__(split,  data_root, dataset, mode, transform_dict)