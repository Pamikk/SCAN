
import numpy as np
import random
import json
dataset = 'ICDAR'
#Train Setting
class Config:
    def __init__(self,mode='train'):
        #Path Setting
        self.img_path = f'../dataset/{dataset}/train' 
        self.checkpoint='../checkpoints'#path to save model weights
        self.dictionary = json.load(open(f'./data/dictionary_{dataset}.json','r'))   
        self.bs = 8
        self.exp = 'exp' #default experiment name
        #data Setting
        self.width = 256
        self.height = 32
        self.windows = [24,32,40]
        self.step = 4
        self.cls_num = len(self.dictionary)       
        if mode=='train':
            self.file=f'./data/train_{dataset}.json'
            self.bs = 32 # batch size
            self.step = 8
            
            #augmentation parameter
            self.rot = 10
            self.scale = 0.1
            self.shear = 0.1
            self.valid_scale = 0.25
            #train_setting
            self.lr = 0.1
            self.weight_decay=5e-4
            self.momentum = 0.9
            #lr_scheduler
            self.min_lr = 5e-5
            self.lr_factor = 0.25
            self.patience = 12
            #exp_setting
            self.save_every_k_epoch = 15
            self.val_every_k_epoch = 10
            self.adjust_lr = False
            self.fine_tune = False


        elif mode=='val':
            self.file = f'./data/test_{dataset}.json'
        elif mode=='trainval':
            self.file = f'./data/trainval_{dataset}.json'
        elif mode=='test':
            self.file = f'./data/test_{dataset}.json'
        
