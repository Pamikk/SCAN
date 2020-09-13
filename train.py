###libs
import os
import argparse
import torch
from torch.utils.data import DataLoader
###files
from config import Config as cfg
from dataProcessing import OCR_dataset as dataset
from models.network import NetAPI
from trainer import Trainer
import warnings
from loss_funcs import LossAPI

warnings.filterwarnings('ignore')
def main(args,cfgs):
    #get data config
    config  = cfgs['train']
    val_cfg = cfgs['val']
    trainval_cfg = cfgs['trainval']
    test_cfg = cfgs['test']
    train_set = dataset(config)
    val_set = dataset(val_cfg,mode='val')
    trainval_set = dataset(trainval_cfg,mode='val')
    test_set = dataset(test_cfg,mode='test')
    if args.bs:
        config.bs = args.bs
    train_loader = DataLoader(train_set,batch_size=config.bs,shuffle=True,pin_memory=False,collate_fn=train_set.collate_fn)
    val_loader = DataLoader(val_set,batch_size=val_cfg.bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    trainval_loader = DataLoader(trainval_set,batch_size=trainval_cfg.bs,shuffle=False,pin_memory=False,collate_fn=val_set.collate_fn)
    test_loader = DataLoader(test_set,batch_size=test_cfg.bs,shuffle=False,pin_memory=False)
    datasets = {'train':train_loader,'val':val_loader,'trainval':trainval_loader,'test':test_loader}
    config.exp_name = args.exp
    config.device = torch.device("cuda")
    torch.cuda.empty_cache()
    if args.lr:
        config.lr = args.lr 
        config.adjust_lr = True
    network = NetAPI(config,args.net)
    loss = LossAPI(config,args.loss)
    torch.cuda.empty_cache()
    det = Trainer(config,datasets,network,loss,(args.resume,args.epochs))
    if args.mode=='val':
        metrics = det.validate(det.start-1,mode='val')        
        det.logger.write_metrics(det.start-1,metrics,[])
        metrics = det.validate(det.start-1,mode='train')
        det.logger.write_metrics(det.start-1,metrics,[],mode='Trainval')
    elif args.mode=='test':
        det.test()
    else:
        det.train()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--resume", type=str, default=None, help="start from epoch?")
    parser.add_argument("--exp",type=str,default='exp',help="name of exp")
    parser.add_argument("--mode",type=str,default='train',help="only validation")
    parser.add_argument("--loss",type=str,default='ctc',help="loss type")
    parser.add_argument("--net",type=str,default='SCCM',help="network type:SCCM|SCAN")
    parser.add_argument("--bs",type=int,default=None,help="batchsize")
    parser.add_argument("--lr",type=float,default=None)
    args = parser.parse_args()
    cfgs = {}
    cfgs['train'] = cfg()
    cfgs['trainval'] = cfg('trainval')
    cfgs['val'] = cfg('val')
    cfgs['test'] = cfg('test')
    main(args,cfgs)
    
    

    