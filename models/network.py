import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .backbone import Basenet
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
def NetAPI(cfg,net,init=True):
    networks = {'SCCM':SCCM}
    network = networks[net](cfg)
    if init:
        network.apply(init_weights)
    return network
def naive_decode(pred):
    seq = torch.argmax(pred,dim=1)
    i=0
    out=[]
    for i,char in enumerate(seq):
        if out:
            if (char!=0) and (char!=seq[i-1]):
                out.append(char.item())
        elif char!=0:
            out.append(char.item())
    return out
class SCCM(nn.Module):
    def __init__(self,cfg):
        super(SCCM,self).__init__()
        self.encoder = Basenet(len(cfg.windows))
        size = cfg.height//(2**self.encoder.depth)
        in_channel = self.encoder.channel*size*size
        self.relu = nn.LeakyReLU(0.01)
        self.pred = nn.Sequential(nn.Linear(in_channel,900),self.relu,
                                nn.Dropout(p=0.5),
                                nn.Linear(900,200),self.relu,
                                nn.Linear(200,cfg.cls_num+1))
    def single_encode(self,x):
        feat = self.encoder(x)
        feat = torch.flatten(feat,start_dim=1)
        out = self.pred(feat)
        return F.log_softmax(out,dim=1)
    def decode(self,result):
        seqs = []
        result = result.permute(1,0,2)
        for i in range(result.shape[0]):
            seqs.append(naive_decode(result[i]))
        return seqs
    def forward(self,x):
        #Input:Bn,Pn,Wn,height,height
        #Pn = num of Image Patches
        Pn = x.shape[1]
        res=[]
        for i in range(Pn):
            res.append(self.single_encode(x[:,i,:,:,:]))
        result = torch.stack(res)
        #shape:Bn,Pn,Cn
        if self.training:
            return result
        else:
            return self.decode(result)
    


    




    




    