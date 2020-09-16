import os
import numpy as np
import scipy.io as sio
import json
def add_key_to_dict(key,dicty):
    idx = len(dicty)
    dicty[key] = idx
    return dicty

def load_data_mat(path,name,keys):
    data = sio.loadmat(os.path.join(path,name+'.mat'))[name][0]
    num = data.shape[0]
    annos = []
    for i in range(num):
        word = data[i]['GroundTruth'][0]
        image = data[i]['ImgName'][0]
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
    path = '../../dataset/IIIT5K'
    anno,keys = load_data_mat(path,'testdata',[])
    json.dump(anno,open('test_IIIT5k.json','w'))
    dictionary =json.load(open('dictionary_IIIT5k.json','r'))
    for key in keys:
        if key not in dictionary.keys():
            print(key,len(dictionary))
            dictionary = add_key_to_dict(key,dictionary)
    json.dump(dictionary,open('dictionary_IIIT5k.json','w'))
def main():
    path = '../../dataset/IIIT5K'
    anno,keys = load_data_mat(path,'traindata',[])
    dictionary = dict([(key,i) for i,key in enumerate(sorted(keys))])
    print(len(dictionary))
    json.dump(anno,open('train_IIIT5k.json','w'))
    json.dump(dictionary,open('dictionary_IIIT5k.json','w'))
if __name__ == '__main__':
    main()
    load_test()