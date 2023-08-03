from modelClasses import *
from data import *
import argparse
import numpy as np
from torch import nn
import time
import os
from trainerClasses import *

import wandb
import torch

parser = argparse.ArgumentParser(description="DEQ-MPI Learned Consistency Pre-Training")
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

parser.add_argument("--wandbFlag", type=int, default=0, help = "use wandb = 1 for tracking loss")

parser.add_argument("--fixedNsStdFlag", type=int, default=1, help= '0: randomly generate noise std for each image, 1: fix noise std.')
parser.add_argument("--pSNRdataList", type=str, default='40',
                    help='input pSNR, separate with comma for training of multiple different networks')

parser.add_argument("--batch_size_trainInpList", type=str, default="64",
                    help="Batch Size")
parser.add_argument("--epoch_nb", type=int, default=150,
                    help="Number of Epochs")

parser.add_argument("--wandbName", type=str,
                    default="deqmpiDC", help='experiment name for WANDB')
parser.add_argument("--mtxCode", type=str, default="./inhouseData/expMatinHouse.mat",
                    help='.mat file path of system matrix')

parser.add_argument("--nbOfSingulars", type=int,
                    default=250, help="Nb of singular values used for least squares initialization")
parser.add_argument("--reScaleBetween", type=str, default="1,1",
                    help='scale images randomly between')

parser.add_argument("--reScaleEpsilon", type=float, default=1, help="rescale epsilon value in inference by. Higher scaling may help improve performance for high pSNR")
parser.add_argument("--lrEpoch", type=int, default=150, help='Update learning rate every X epoch')

parser.add_argument("--useDCNormalization", type=int, default=0, help='Use Data Consistency Normalization Type: 0: No normalization: 1 proposed normalization')

parser.add_argument("--nb_of_featuresLList", type=str,
                    default='8', help='Number of features of learned consistency network, separate with comma for training of multiple different networks')
parser.add_argument("--nb_of_blocksLList", type=str,
                    default='1', help='Number of blocks of learned consistency network, separate with comma for training of multiple different networks')

parser.add_argument("--noisySysMtx", type = float, default = 0.1, help = 'Add noise to sys mtx')
parser.add_argument("--consistencyDim", type = int, default = 1, help = 'Dimensionality of the learned data consistency: 0: conventional consistency, 1: 1D consistency')
parser.add_argument("--useLoss", type = int, default = 0, help = '0: l1, 1: l2')


opt = parser.parse_args()
print(opt)

useGPUno = opt.useGPU
torch.cuda.set_device(useGPUno)
batch_size_trainInpList = np.array(
    opt.batch_size_trainInpList.split(',')).astype(int)
batch_size_val = 4096
epoch_nb = opt.epoch_nb

saveModelEpoch = opt.saveModelEpoch
wandbFlag = bool(opt.wandbFlag)
pSNRdataList = np.array(opt.pSNRdataList.split(',')).astype(float)
valEpoch = opt.valEpoch
wandbProjectName = opt.wandbName
fixedNsStdFlag = bool(opt.fixedNsStdFlag)
mtxCode = opt.mtxCode
nbOfSingulars = opt.nbOfSingulars
reScaleEpsilon = opt.reScaleEpsilon

reScaleBetween = np.array(opt.reScaleBetween.split(",")).astype(float)

reScaleMin = reScaleBetween[0]
reScaleMax = reScaleBetween[1] - reScaleBetween[0]

nb_of_featuresLList = np.array(opt.nb_of_featuresLList.split(',')).astype(int)
nb_of_blocksLList = np.array(opt.nb_of_blocksLList.split(',')).astype(int)

lr = opt.lr
weight_decay = opt.wd

#
resultFolder = "training/dcDenoiser"

Ul = list()
Sl = list()
Vl = list()
Sysl = list()
theSysl = list()

n1 = 26
n2 = 13
cnt = 1
dims = 2
theImgSizes = [n1, n2]

for mtcesName in range(cnt):
    sysMtx = loadMtxExp(mtxCode).reshape(-1, n1 * n2)

    U, S, Vh = torch.linalg.svd(
        sysMtx, full_matrices=False)

    nbSvd = nbOfSingulars
    U_ = U[:, :nbSvd]
    S_ = S[:nbSvd]
    Vh_ = Vh[:nbSvd, :]
    V_ = Vh_.T

    theSys = U_.T @ sysMtx

    Ul.append(U_)
    Sl.append(S_)
    Vl.append(V_)
    theSysl.append(theSys)
    Sysl.append(sysMtx)

input_channels = 1

mraFolderPath = "./datasets/"

tmpTm3 = time.time()
trainDataset = MRAdatasetH5NoScale(mraFolderPath + "trainPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds for train set to be moved to RAM'.format(time.time()-tmpTm3)) # myflag
print('Train set size:',trainDataset.__len__())
tmpTm4 = time.time()
valDataset = MRAdatasetH5NoScale(mraFolderPath + "valPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds for validation set to be moved RAM'.format(time.time()-tmpTm4)) # myflag
print('Validation set size:',valDataset.__len__())

def callMyFnc(nb_of_featuresL, nb_of_blocksL, lr, batch_size_train, weight_decay, pSNRval):
    consistencyDim = opt.consistencyDim
    tempStr = 'dcDenoiserPsi' + "_" + str(consistencyDim) + "D_ds_lr_"+str(lr)+"_wd_"+str(weight_decay)+"_bs_"\
        + str(batch_size_train)+"_pSNR_"+str(pSNRval)+"_fixNs_"+str(int(fixedNsStdFlag))\
        + "_rMn" + str(reScaleMin) + \
        "_" + str(reScaleMax)

    tempStr = tempStr+ '_mtx_'+mtxCode[-15:] + \
        '_svd_'+str(nbOfSingulars) + \
        '_LnF_' + str(nb_of_featuresL) + \
        '_LnB_' + str(nb_of_blocksL) + "_nN_" + str(opt.useDCNormalization) + \
        '_sN_' + str(opt.noisySysMtx) + '_ls_' + str(opt.useLoss)

    saveFolder = resultFolder + "/" + tempStr

    if wandbFlag:
        wandb.init(project=wandbProjectName,
                    reinit=True, name=tempStr, tags = ['{0}std'.format(opt.noisySysMtx)])

    print(opt)

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    if consistencyDim == 0:
        model = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon)
    elif consistencyDim < 4:
        model = consistencyNetworkMD(nb_of_featuresL, nb_of_blocksL, useNormalization = opt.useDCNormalization, numDim = consistencyDim).cuda()

    torch.autograd.set_detect_anomaly(True)
    print("num params: ", sum(p.numel()
                                for p in model.parameters() if p.requires_grad))
    
    lrUpdateEpoch = epoch_nb // 5 if opt.lrEpoch == 0 else opt.lrEpoch // 5

    if opt.useLoss == 0:
        loss = nn.L1Loss().cuda()
    else:
        loss = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lrUpdateEpoch, gamma=0.5)

    model, trainMetrics, valMetrics = trainDCdenoiserPsi(model=model,
                                                epoch_nb=epoch_nb,
                                                loss=loss,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                trainDataset=trainDataset,
                                                valDataset=valDataset,
                                                batch_size_train=batch_size_train,
                                                batch_size_val=batch_size_val,
                                                theSysl=theSysl,
                                                sysMtxl=Sysl,
                                                Ul=Ul,
                                                Sl=Sl,
                                                Vl=Vl,
                                                noisySys = opt.noisySysMtx,
                                                imgSizes=theImgSizes,
                                                rescaleVals=[
                                                    reScaleMin, reScaleMax],
                                                saveModelEpoch=saveModelEpoch,
                                                valEpoch=valEpoch,
                                                saveDirectory=saveFolder,
                                                pSNRval=pSNRval,
                                                wandbFlag=wandbFlag,
                                                fixedNoiseStdFlag=fixedNsStdFlag,
                                                nbOfSingulars=nbOfSingulars,
                                                lambdaVal=reScaleEpsilon)

for nb_of_featuresL in nb_of_featuresLList:# = np.array(opt.nb_of_featuresLList.split(',')).astype(int)
    for nb_of_blocksL in nb_of_blocksLList:
        for batch_size_train in batch_size_trainInpList:
            batch_size_train = int(batch_size_train)
            for pSNRval in pSNRdataList:
                callMyFnc(nb_of_featuresL, nb_of_blocksL, lr, batch_size_train, weight_decay, pSNRval)
