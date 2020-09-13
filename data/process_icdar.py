import os
import cv2
import numpy as np
import scipy.io as sio
import json
def add_key_to_dict(key,dicty):
    idx = len(dicty)
    dicty[key] = idx
    return dicty

def parse_line(data):
    start = data.find("\"")
    end = data[start+1:].find("\"")
    gt = data[start+1:start+1+end]
    name = data[:start].strip().strip(',').strip()
    return name,gt
def load_data_mat_txt(path,name,keys):
    datas = list(open(os.path.join(path,name+'.txt'),'r').readlines())
    num = len(datas)
    annos = []
    for i in range(num):
        image,word = parse_line(datas[i])
        anno = {}
        anno['img_id'] = i
        anno['img_name'] = image
        anno['ground_truth'] = word
        for char in word:
            if char not in keys:
                keys.append(char)
        annos.append(anno)
    return annos,keys
def load_test():
    path = '../../dataset/ICDAR/test'
    anno,keys = load_data_mat_txt(path,'gt',[])
    json.dump(anno,open('test_ICDAR.json','w'))
    dictionary =json.load(open('dictionary_ICDAR.json','r'))
    for key in keys:
        if key not in dictionary.keys():
            print(key,len(dictionary))
            dictionary = add_key_to_dict(key,dictionary)
    json.dump(dictionary,open('dictionary_ICDAR.json','w'))
def main():
    path = '../../dataset/ICDAR/train'
    anno,keys = load_data_mat_txt(path,'gt',[])
    dictionary = dict([(key,i) for i,key in enumerate(sorted(keys))])
    json.dump(anno,open('train_ICDAR.json','w'))
    json.dump(dictionary,open('dictionary_ICDAR.json','w'))
    







if __name__ == '__main__':
    load_test()