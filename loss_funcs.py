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
        lens = torch.full(size=(bn,), fill_value=0, dtype=torch.long)
        for i in range(bn):
            lens[i] = torch.sum(gt[:,0] == i).long()
        return lens.cuda()
    def forward(self,outs,gt=None,infer=False):
        if infer:
            return outs
        else:
            pn,bn,_ = outs.shape
            tlens = self.get_tlength(gt,bn)
            plens = torch.tensor([pn]*bn,dtype=torch.long).cuda()
            gt = gt[:,1].long()
            val = self.loss(outs,gt,plens,tlens)
            return {'all':val},val

        




        







        