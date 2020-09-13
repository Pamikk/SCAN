import matplotlib.pyplot as plt 
import math
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os 
import json
from tqdm import tqdm

class Logger(object):
    def __init__(self,log_dir):
        self.log_dir = log_dir
        self.files = {'val':open(os.path.join(log_dir,'val.txt'),'a+'),'train':open(os.path.join(log_dir,'train.txt'),'a+')}
    def write_line2file(self,mode,string):
        self.files[mode].write(string+'\n')
        self.files[mode].flush()
    def write_loss(self,epoch,losses,lr):
        tmp = str(epoch)+'\t'+str(lr)+'\t'
        print('Epoch',':',epoch,'-',lr)
        writer = SummaryWriter(log_dir=self.log_dir)
        writer.add_scalar('lr',math.log(lr),epoch)
        for k in losses:
            if losses[k]>0:            
                writer.add_scalar('Train/'+k,losses[k],epoch)            
                print(k,':',losses[k])
                #self.writer.flush()
        tmp+= str(round(losses['all'],5))+'\t'
        self.write_line2file('train',tmp)
        writer.close()
    def write_metrics(self,epoch,metrics,save=[],mode='Val',log=True):
        tmp =str(epoch)+'\t'
        print("validation epoch:",epoch)
        writer = SummaryWriter(log_dir=self.log_dir)
        for k in metrics:
            if k in save:
                tmp +=str(metrics[k])+'\t'
            if log:
                tag = mode+'/'+k            
                writer.add_scalar(tag,metrics[k],epoch)
                #self.writer.flush()
            print(k,':',metrics[k])
        
        self.write_line2file('val',tmp)
        writer.close()
def get_sentence(pred,dictionary):
    conv = dict([(dictionary[k],k) for k in dictionary])
    return str.join('',[conv[i] for i in pred])
def cal_correct_num(pred,label):
    cor = 0
    for pd,gt in zip(pred,label):
        if pd==gt:
            cor +=1
    return cor











