import wandb
import torch
import sys

from torch.utils.data import Dataset, DataLoader
from data import *
from modelClasses import *
from trainerClasses import *

import os

import time
import torch
from torch import nn
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Rdn grid search")

parser.add_argument("--useGPU", type=int, default=0,
                    help="GPU ID to be utilized")

parser.add_argument("--wd", type=float, default=0,
                    help='weight decay')

parser.add_argument("--lr", type=float,
                    default=1e-3, help='learning rate')

parser.add_argument("--saveModelEpoch", type=int,
                    default=99, help="Save model per epoch")

parser.add_argument("--valEpoch", type=int, default=10,
                    help="compute validation per epoch")


parser.add_argument("--fixedNsStdFlag", type=int, default=1, help= '0: randomly generate noise std for each image, 1: fix noise std.')

parser.add_argument("--minNoiseStd", type=float, default=0, help='For non-fixed noise, minimum noise std.')

parser.add_argument("--maxNoiseStdList", type=str, default='0.1', help='For non-fixed noise: maximum noise std., For fixed noise: noise std.')

parser.add_argument("--batch_size_train", type=int, default=64,
                    help="Batch Size")

parser.add_argument("--epoch_nb", type=int, default=150,
                    help="When to decay learning rate; should be less than epochs")

parser.add_argument("--wandbFlag", type=int, default=0, help = "use wandb = 1 for tracking loss")
parser.add_argument("--wandbName", type=str,
                    default="ppmpi", help='experiment name for WANDB')

parser.add_argument("--reScaleBetween", type=str, default="1,1",
                    help='scale images randomly between')

parser.add_argument("--dims", type=int, default=2, help='Number of dimensions of the denoiser')


parser.add_argument("--nb_of_featuresList", type=str,
                    default="12", help='Number of features of RDN, separate with comma for training of multiple different networks')
parser.add_argument("--nb_of_blocks", type=int,
                    default=4, help='Number of blocks of RDN')
parser.add_argument("--layer_in_each_block", type=int,
                    default=4, help='Layer in each block of RDN')
parser.add_argument("--growth_rate", type=int, default=12,
                    help='growth rate of RDN')


opt = parser.parse_args()
print(opt)

asel = True
dims = opt.dims
resultFolder = "training/denoiser" if dims == 2 else "training/denoiser3d" 

useGPUno = opt.useGPU
torch.cuda.set_device(useGPUno)

batch_size_train = opt.batch_size_train
weight_decay = opt.wd
lr = opt.lr
layer_in_each_block = opt.layer_in_each_block
nb_of_blocks = opt.nb_of_blocks
growth_rate = opt.growth_rate
nb_of_featuresList = np.array(opt.nb_of_featuresList.split(',')).astype(int)

batch_size_val = 4096
epoch_nb = opt.epoch_nb
saveModelEpoch = opt.saveModelEpoch
wandbFlag = bool(opt.wandbFlag)
minNoiseStd = opt.minNoiseStd
maxNoiseStdList = np.array(opt.maxNoiseStdList.split(',')).astype(float)
valEpoch = opt.valEpoch
wandbProjectName = opt.wandbName
fixedNsStdFlag = bool(opt.fixedNsStdFlag)

mraFolderPath = "datasets/"

reScaleBetween = np.array(opt.reScaleBetween.split(",")).astype(float)

reScaleMin = reScaleBetween[0]
reScaleMax = reScaleBetween[1] - reScaleBetween[0]

tmpTm3 = time.time()
trainDataset = MRAdatasetH5NoScale(mraFolderPath + "trainPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds from train set to RAM'.format(time.time()-tmpTm3)) # myflag
print('Train set size:',trainDataset.__len__())
tmpTm4 = time.time()
valDataset = MRAdatasetH5NoScale(mraFolderPath + "valPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds from val set to GPU'.format(time.time()-tmpTm4)) # myflag
print('Validation set size:',valDataset.__len__())

for nb_of_features in nb_of_featuresList:
    for maxNoiseStd in maxNoiseStdList:
        tempStr = "ppmpi_lr_"+str(lr)+"_wd_"+str(weight_decay)+"_bs_"\
            + str(batch_size_train)+"_mxNs_"+str(maxNoiseStd)+"_fixNs_"+str(int(fixedNsStdFlag))\
            +"_data_mnNs_"+str(minNoiseStd)\
            + '_nF'+str(nb_of_features)+'_nB'+str(nb_of_blocks)\
            + '_lieb'+str(layer_in_each_block) + \
            '_gr'+str(growth_rate) + \
            "_rMn" + str(reScaleMin) + \
            "_" + str(reScaleMax)

        tempStr = tempStr if dims == 2 else tempStr + "_3d"
        
        saveFolder = resultFolder + "/"+ tempStr
        optionalMessage=""

        if wandbFlag:
            wandb.init(project=wandbProjectName, reinit=True, name=tempStr)

        print(opt)
        print(optionalMessage)

        if not os.path.exists(saveFolder):
            os.makedirs(saveFolder)

        model = rdnDenoiserResRelu(
            1, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, 1, bias=True).cuda() if dims == 2 else rdnDenoiserResRelu3d(1, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, 1, bias=True).cuda() 

        print("number of trainable parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
        loss = nn.L1Loss().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_nb // 5, gamma=0.5)

        model, trainMetrics, valMetrics = trainDenoiser(model = model,
                                        epoch_nb = epoch_nb,
                                        loss = loss,
                                        optimizer = optimizer,
                                        scheduler = scheduler,
                                        trainDataset = trainDataset,
                                        valDataset = valDataset,
                                        batch_size_train = batch_size_train,
                                        batch_size_val = batch_size_val,
                                        rescaleVals=[reScaleMin, reScaleMax],
                                        saveModelEpoch=saveModelEpoch,
                                        valEpoch=valEpoch,
                                        saveDirectory=saveFolder, 
                                        maxNoiseStd = maxNoiseStd,
                                        optionalMessage=optionalMessage,
                                        wandbFlag = wandbFlag,
                                        fixedNoiseStdFlag = fixedNsStdFlag,
                                        minNoiseStd = minNoiseStd,
                                        dims = dims)