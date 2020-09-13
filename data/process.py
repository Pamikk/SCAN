import json
import os
import random

def train_val_split(anno,ratio=0.2,num=200):
    random.shuffle(anno)
    n = len(anno)
    val = anno[:int(n*ratio)]
    train = anno[int(n*ratio):]
    trainval = train[:min(num,int(len(train)*ratio))]
    print(len(val),len(train),len(trainval))
    return val,train,trainval

dataset ='ICDAR' #IIIT5k
anno = json.load(open(f'annotations_{dataset}.json','r'))

val,train,trainval = train_val_split(anno)
json.dump(val,open(f'val_{dataset}.json','w'))
json.dump(train,open(f'train_{dataset}.json','w'))
json.dump(trainval,open(f'trainval_{dataset}.json','w'))
