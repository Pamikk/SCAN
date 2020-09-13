import json
import numpy as np
import os
import cv2
from tqdm import tqdm
def analyze_size(path,annos):
    mh,mw = 1000,1000
    mxh,mxw = 0,0
    gray = 0
    res = {}
    res2 = {}
    for anno in tqdm(annos):   
        name = anno['img_name']
        img = cv2.imread(os.path.join(path,name),0)
        if len(img.shape)==2:
            gray+=1
            h,w =img.shape
        else:
            h,w,_ = img.shape
        if h>w:
            if int(h/w) in res.keys():
                res[int(h/w)].append((name,h,w))
            else:
                res[int(h/w)]=[(name,h,w)]
        else:
            if int(w/h) in res2.keys():
                res2[int(w/h)].append((name,h,w))
            else:
                res2[int(w/h)]=[(name,h,w)] 
        mh = min(mh,h)
        mw = min(mw,w)
        mxh = max(mxh,h)
        mxw = max(mxw,w)
    json.dump(res,open('result.json','w'))
    for i in res:
        print(i,len(res[i]))
    for i in res2:
        print(i,len(res2[i]))
    print(gray)
    print('h:',mh,mxh)
    print('w:',mw,mxw)

if __name__ == "__main__":
    dataset = 'ICDAR'
    path = f'../dataset/{dataset}/train'
    anno = json.load(open(f'./data/train_{dataset}.json','r'))
    analyze_size(path,anno)