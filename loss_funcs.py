import torch.nn as nn
import torch
import numpy as np

ctcloss = nn.CTCLoss()

class LossAPI(nn.Module):
    def __init__(self,cfg,loss):
        super(LossAPI,self).__init__()
        Losses ={'ctc':ctcloss}
        self.loss = Losses[loss]
    def get_tlength(self,gt,bn):
        lens = list(range(bn))
        for i in range(bn):
            lens[i] = torch.sum(gt[:,0] == i).long()
        return lens
    def forward(self,outs,gt=None):
        if gt:
            bn,pn,_ = outs.shape
            tlens = self.get_tlength(gt,bn)
            plens = torch.full(size=(bn,), fill_value=pn, dtype=torch.long)
            val = self.loss(outs,gt[:,1],plens,tlens)
            return {'all':val},val
        else:
            return outs

        




        







        