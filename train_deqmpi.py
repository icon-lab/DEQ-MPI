
import wandb
import torch
import argparse
import numpy as np
import time
import os

from torch import nn
from modelClasses import *
from data import *
from trainerClasses import *


parser = argparse.ArgumentParser(description="DEQ-MPI Training")
parser.add_argument("--useGPU", type=int, default=0,
                    help="GPU ID to be utilized")

# Training & Optimizer Parameters
parser.add_argument("--wd", type=float, default=0,
                    help='weight decay')
parser.add_argument("--lr", type=float,
                    default=1e-3, help='learning rate')
parser.add_argument("--lrEpoch", type=int, default=150, help='Update learning rate every X epoch')

parser.add_argument("--saveModelEpoch", type=int,
                    default=99, help="Save model per epoch")
parser.add_argument("--valEpoch", type=int, default=10,
                    help="compute validation per epoch")
parser.add_argument("--batch_size_train", type=int, default=64,
                    help="Batch Size")
parser.add_argument("--epoch_nb", type=int, default=150,
                    help="Number of Epochs")

parser.add_argument("--wandbFlag", type=int, default=0, help = "use wandb = 1 for tracking loss")
parser.add_argument("--wandbName", type=str,
                    default="deqmpi", help='experiment name for WANDB')
parser.add_argument("--optionalString", type = str, default = "", help = 'Optional Naming for WanDB and saving model')

# Training Forward Model Options
parser.add_argument("--fixedNsStdFlag", type=int, default=1, help= '0: randomly generate noise std for each image, 1: fix noise std.')
parser.add_argument("--pSNRdataList", type=str, default='40',
                    help='input pSNR, separate with comma for training of multiple different networks')

parser.add_argument("--mtxCode", type=str, default="./inhouseData/expMatinHouse.mat",
                    help='.mat file path of system matrix')
parser.add_argument("--nbOfSingulars", type=int,
                    default=250, help="Nb of singular values used for least squares initialization")

# Data Processing Options

parser.add_argument("--reScaleBetween", type=str, default="1,1",
                    help='scale images randomly between')
parser.add_argument("--reScaleEpsilon", type=float, default=1, help="rescale epsilon value in inference by. Higher scaling may help improve performance for high pSNR")


# Model options

parser.add_argument("--modelType", type=str, default="DeqMPI", help="model type: ADMLD (Unrolled) or DeqMPI")

parser.add_argument("--nb_of_featuresList", type=str,
                    default="12", help='Number of features of RDN, separate with comma for training of multiple different networks')
parser.add_argument("--nb_of_blocks", type=int,
                    default=4, help='Number of blocks of RDN')
parser.add_argument("--layer_in_each_block", type=int,
                    default=4, help='Layer in each block of RDN')
parser.add_argument("--growth_rate", type=int, default=12,
                    help='growth rate of RDN')
parser.add_argument("--consistencyDimList", type = str, default = "1", help = 'Dimensionality of the learned data consistency: 0: conventional consistency, 1: 1D consistency')

parser.add_argument("--nb_of_steps", type=int, default=5,
                    help='Number of steps. ONLY used for unrolled variant')

parser.add_argument("--nb_of_featuresLList", type=str,
                    default='8', help='Number of features of learned consistency network, separate with comma for training of multiple different networks')
parser.add_argument("--nb_of_blocksL", type=int,
                    default=1, help='Number of blocks of learned consistency network')
parser.add_argument("--useDCNormalization", type=int, default=1, help='Use Data Consistency Normalization Type: 0: No normalization: 1 proposed normalization')

parser.add_argument("--preLoadDir", type=str, default="", help="preload denoiser network path")
parser.add_argument("--preLoadDirDC", type=str, default="", help="preload learned consistency network path")


torch.autograd.set_detect_anomaly(True)

opt = parser.parse_args()
print(opt)

useGPUno = opt.useGPU
torch.cuda.set_device(useGPUno)
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
preLoadDir = opt.preLoadDir
preLoadDirDC = opt.preLoadDirDC

reScaleBetween = np.array(opt.reScaleBetween.split(",")).astype(float)

reScaleMin = reScaleBetween[0]
reScaleMax = reScaleBetween[1] - reScaleBetween[0]

nb_of_featuresLList = np.array(opt.nb_of_featuresLList.split(',')).astype(int)
nb_of_featuresList = np.array(opt.nb_of_featuresList.split(',')).astype(int)
consistencyDimList = np.array(opt.consistencyDimList.split(',')).astype(int)

lr = opt.lr
weight_decay = opt.wd
batch_size_train = opt.batch_size_train
nb_of_steps = opt.nb_of_steps
nb_of_blocks = opt.nb_of_blocks
layer_in_each_block = opt.layer_in_each_block
growth_rate = opt.growth_rate
nb_of_blocksL = opt.nb_of_blocksL

#

if opt.modelType == "ADMLD":
    resultFolder = "training/admld"
    trainType = 3
elif opt.modelType == "DeqMPI":
    resultFolder = "training/deqmpi"
    trainType = 2

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

mraFolderPath = "./datasets/"

tmpTm3 = time.time()
trainDataset = MRAdatasetH5NoScale(mraFolderPath + "trainPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds for train set to be moved to RAM'.format(time.time()-tmpTm3)) # myflag
print('Train set size:',trainDataset.__len__())

tmpTm4 = time.time()
valDataset = MRAdatasetH5NoScale(mraFolderPath + "valPatches.h5",prefetch=True, dim = dims, device = torch.device('cpu'))
print('It takes {0:.2f} seconds for validation set to be moved RAM'.format(time.time()-tmpTm4)) # myflag
print('Validation set size:',valDataset.__len__())


def callMyFnc(nb_of_featuresL, nb_of_blocksL, growth_rate, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, lr, batch_size_train, weight_decay, pSNRval, consistencyDim):
    tempStr = opt.modelType + opt.optionalString + "_" + str(consistencyDim) + "D_ds_lr_"+str(lr)+"_wd_"+str(weight_decay)+"_bs_"\
        + str(batch_size_train)+"_pSNR_"+str(pSNRval)+"_fixNs_"+str(int(fixedNsStdFlag))\
        + '_Nit_'+str(nb_of_steps)+'_nF'+str(nb_of_features)+'_nB'+str(nb_of_blocks)\
        + '_lieb'+str(layer_in_each_block) + \
        '_gr'+str(growth_rate) + \
        "_rMn" + str(reScaleMin) + \
        "_" + str(reScaleMax) + \
        '_mtx_' + mtxCode[-15:] + \
        '_svd_' + str(nbOfSingulars) + \
        '_LnF_' + str(nb_of_featuresL) + \
        '_LnB_' + str(nb_of_blocksL) + "_nN_" + str(opt.useDCNormalization)
    

    saveFolder = resultFolder + "/" + tempStr

    if wandbFlag:
        wandb.init(project=wandbProjectName,
                    reinit=True, name=tempStr, tags = [opt.modelType, "{0}dB".format(pSNRval), "DCN={0}".format(opt.useDCNormalization), 
                    "nCD={0}".format(consistencyDim)])

    print(opt)

    if not os.path.exists(saveFolder):
        os.makedirs(saveFolder)

    Ml2 = list()

    for ii in range(cnt):
        ssL = theSysl[ii]
        M_ = torch.inverse(torch.eye(ssL.shape[1]).type_as(
            ssL).to(ssL.device) + ssL.T @ ssL)
        Ml2.append(M_)

    useNormalizationL = opt.useDCNormalization
    if opt.modelType == "ADMLD":
        # nb_of_featuresL, nb_of_blocksL = 8, 1 
        model = rdnADMMLDnet(
            1, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL, bias=True, numDim=dims, consistencyDim = consistencyDim).cuda()
        print(model)

    elif opt.modelType == "DeqMPI":
        model2 = rdnLDFixedPt(
            1, nb_of_features, nb_of_blocks, layer_in_each_block, growth_rate, nb_of_featuresL, nb_of_blocksL, useNormalizationL, bias=True, numDim=dims, consistencyDim = consistencyDim).cuda()
        
        if preLoadDir != "":
            model2.sharedNet.load_state_dict(torch.load(preLoadDir, map_location=next(model2.parameters()).device))
        if preLoadDirDC != "":
            model2.consistencyNet.load_state_dict(torch.load(preLoadDirDC, map_location=next(model2.parameters()).device))

        model = DEQFixedPoint(model2, anderson, tol = 1e-4, max_iter = 25, beta = 2.0)

    print("num params: ", sum(p.numel()
                                for p in model.parameters() if p.requires_grad))
    
    lrUpdateEpoch = epoch_nb // 5 if opt.lrEpoch == 0 else opt.lrEpoch // 5

    loss = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lrUpdateEpoch, gamma=0.5)

    model, trainMetrics, valMetrics = trainADMMandE2EandImplicit(model=model,
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
                                                Ml=Ml2,
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
                                                lambdaVal=reScaleEpsilon,
                                                mode = trainType)


for consistencyDim in consistencyDimList:
    for nb_of_featuresL in nb_of_featuresLList:
        for nb_of_features in nb_of_featuresList:
            for pSNRval in pSNRdataList:
                callMyFnc(nb_of_featuresL, nb_of_blocksL, growth_rate, nb_of_steps, nb_of_features, nb_of_blocks, layer_in_each_block, lr, batch_size_train, weight_decay, pSNRval, consistencyDim)
