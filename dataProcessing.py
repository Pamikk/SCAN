import torch.utils.data as data
import torch
import json
import numpy as np
import random
import cv2
import os
from torch.nn import functional as F

ls = 1 #0 for only bboxes,1 for labels and bboxes
#stack functions for collate_fn
#Notice: all dicts need have same keys and all lists should have same length
def stack_dicts(dicts):
    if len(dicts)==0:
        return None
    res = {}
    for k in dicts[0].keys():
        res[k] = [obj[k] for obj in dicts]
    return res

def stack_list(lists):
    if len(lists)==0:
        return None
    res = list(range(len(lists[0])))
    for k in range(len(lists[0])):
        res[k] = torch.stack([obj[k] for obj in lists])
    return res
def rand(item):
    try:
        tmp=[]
        for i in item:
            tmp.append(random.uniform(-i,i))
    except:
        if random.random()<0.5:
            return random.uniform(-i,i)
        else:
            return 0
    finally:
        return tuple(tmp)   
def get_croppable_part(labels):
    min_x = torch.min(labels[:,ls]-labels[:,ls+2]/2)
    min_y = torch.min(labels[:,ls+1]-labels[:,ls+3]/2)
    max_x = torch.max(labels[:,ls]+labels[:,ls+2]/2)
    max_y = torch.max(labels[:,ls+1]+labels[:,ls+3]/2)
    return (min_x,min_y,max_x,max_y)
def valid_scale(src,vs):
    vs = random.uniform(-vs,vs)
    img = src.astype(np.float)
    img*= (1+vs)
    img[img>255] = 255
    img = img.astype(np.uint8)
    return img
def resize(src,tsize):
    dst = cv2.resize(src,(tsize[1],tsize[0]),interpolation=cv2.INTER_LINEAR)
    return dst
def shear(src,shear):
    h,w = src.shape
    sx = random.uniform(-shear,shear)
    sy = random.uniform(-shear,shear)
    mat = np.array([[1,sx,0],[sy,1,0]])    
    dst = cv2.warpAffine(src,mat,(w,h))
    return dst
def rotate(src,ang,scale):
    h,w = src.shape
    center =(w/2,h/2)
    mat = cv2.getRotationMatrix2D(center, ang, scale)
    dst = cv2.warpAffine(src,mat,(w,h))
    return dst
def color_normalize(img,mean):
    img = img.astype(np.float)
    if img.max()>1:
        img /= 255
    img -= np.array(mean)/255
    return img

class OCR_dataset(data.Dataset):
    def __init__(self,cfg,mode='train'):
        self.img_path = cfg.img_path
        self.cfg = cfg
        data = json.load(open(cfg.file,'r'))
        self.annos = data
        self.mode = mode
        self.accm_batch = 0
        self.width = cfg.width
        self.height = cfg.height
        self.windows = cfg.windows
        self.step = cfg.step
        self.dictionary = cfg.dictionary
    def __len__(self):
        return len(self.annos)

    def img_to_tensor(self,img):
        data = torch.tensor(img,dtype=torch.float)
        if data.max()>1:
             data /= 255.0
        #(width-self.window_size)//self.step,len(window_sizes),self.height,self.height
        return data

    def gen_gts(self,anno):
        labels = anno[ "ground_truth"]
        gt = torch.zeros(len(labels),dtype=torch.long)
        for i,char in enumerate(labels):
            gt[i] = self.dictionary[char] + 1
        return gt

    def slide_images(self,img):
        w = self.width
        assert img.shape[1] == w
        img_patches = []
        max_window_size = max(self.windows)
        for center in range(max_window_size//2,w-max_window_size//2+1,self.step):
            patches = []
            for window_size in self.windows:
                #multiscale
                img_patch = img[:, center - window_size // 2: center + window_size // 2]
                img_patch = cv2.resize(img_patch, (self.height, self.height))
                patches.append(img_patch)
            img_patches.append(np.asarray(patches))
        return np.asarray(img_patches)
    def pre_process_image(self,img):
        #img should be gray-scale
        h,_ = img.shape
        scale = self.height / h
        img = cv2.resize(img, None, fx=scale, fy=scale)
        if img.shape[1] < self.width:
            #keep aspect ratio
            diff = self.width - img.shape[1]
            pad = diff//2
            img = cv2.copyMakeBorder(img,0,0,pad,diff-pad,cv2.BORDER_CONSTANT)
        else:
            img = cv2.resize(img, None, fx=self.width / img.shape[1], fy=1)
        if img.shape[1] != self.width:
            raise ValueError('shape = %d,%d' % img.shape)
        img = img.astype('uint8')
        return img

    def __getitem__(self,idx):
        anno = self.annos[idx]
        name = anno['img_name']
        img_path = os.path.join(self.img_path,name)
        img = cv2.imread(img_path,0)#read as gray
        ##print(img.shape)
        h,w = img.shape[:2]   
        if h>2.5*w:
            img = img.T
        if self.mode=='train':
            if (random.uniform(0,1)>0.25) and self.cfg.shear:
                img = shear(img,self.cfg.shear)
            if (random.uniform(0,1)>0.25) and self.cfg.rot:
                ang = random.uniform(-self.cfg.rot,self.cfg.rot)
                scale = random.uniform(1-self.cfg.scale,1+self.cfg.scale)
                img = rotate(img,ang,scale)
            if (random.uniform(0,1)>0.25) and self.cfg.valid_scale:
                img = valid_scale(img,self.cfg.valid_scale)

        img = self.pre_process_image(img)
        patches = self.slide_images(img)
        data = self.img_to_tensor(patches)
        if self.mode=='test':
            return data,idx
        labels = self.gen_gts(anno)
        return data,labels

    def collate_fn(self,batch):
        data,labels = list(zip(*batch))
        data = torch.stack(data)
        tmp = []                   
                
        for i,label in enumerate(labels):
            if len(label)>0:
                gt= torch.zeros(len(label),2).long()
                gt[:,1] = label
                gt[:,0] = i
                tmp.append(gt)
        if len(tmp)>0:
            labels = torch.cat(tmp,dim=0)
            labels = labels.reshape(-1,2)
        else:
            labels = torch.tensor(tmp,dtype=torch.long).reshape(-1,2)
        return data,labels

                





