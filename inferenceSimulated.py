import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn.functional as F

from data import *
from reconAlgos import *
from modelClasses import *
from trainerClasses import *

import gc
import time
import copy

# Settings for inference:

gpuNo = 0 # use GPU ID

pSNRval = 10 # simulation pSNR


initializationTo = 1 # initialize ADMM to least squares input
# 0: zeros
# 1: least squares input
# 2: Regularized least squares input

# run inference for these techniques:
# L1, TV, L1_TV are the conventional hand-crafted regularizers.
# other ones include folder names under "training/denoiser/", "training/dcDenoiser/", "training/deqmpi/", "training/admld", "training/deqmpi"
# after "+" one can include l1, tv for a linear combination of plug-and-play and l1 and/or tv.

descriptorsHere = [\
"L1",\
"TV",\
"L1_TV",\
"ppmpi_lr_0.001_wd_0_bs_64_mxNs_0.1_fixNs_1_data_mnNs_0_nF12_nB4_lieb4_gr12_rMn0.5_1.0+  ",\
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_15.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_20.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_25.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_30.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_35.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
"DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_40.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
                     ]

descriptorsLD = [\
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_10.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_15.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_20.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_25.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_30.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_35.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
# "DeqMPI_1D_ds_lr_0.001_wd_0.0_bs_64_pSNR_40.0_fixNs_1_Nit_5_nF12_nB4_lieb4_gr12_lmb100.0_rMn0.5_1.0_mtx_pMatinHouse.mat_svd_250_LnF_8_LnB_1_nN_1_cr_0", \
                     ]

nbOfSingulars = 220
inverseCrime = True

inhouseLoadStr = "inhouseData/expMatinHouse"
# inhousePhantomLoadStr = "inhouseData/expPhantominHouse.mat"

torch.cuda.set_device(gpuNo)
print(torch.cuda.get_device_name(gpuNo))

n1 = 26
n2 = 13

sysMtxRef = loadMtxExp(inhouseLoadStr).reshape(-1, n1 * n2)

if inverseCrime: # bicubic upsampling followed by downsampling
    interpolater = loadmat('interpExp2.mat')['interpolater']
    sysMtxHRint2 = sysMtxRef @ torch.from_numpy(interpolater).float().cuda()
    dataGenMtx = sysMtxHRint2
    sysMtx = F.avg_pool2d(sysMtxHRint2.reshape(sysMtxRef.shape[0], 2 * n1, 2 * n2), 2).reshape(sysMtxRef.shape[0], -1)
else:
    dataGenMtx = sysMtxRef
    sysMtx = sysMtxRef

U, S, Vh = torch.linalg.svd(
    sysMtx, full_matrices=False)

nbSvd = nbOfSingulars
U_ = U[:, :nbSvd]
S_ = S[:nbSvd]
Vh_ = Vh[:nbSvd, :]
V_ = Vh_.T
# B = U_.T @ sysMtx

theSys = U_.T @ sysMtx

# Load Phantom Data

# loadPreProcessed = 0


testDataHR = (MRAdatasetH5NoScale("datasets/testPatches.h5", prefetch=True)).data
testData = transformDataset(testDataHR, [26, 13], [0.5, 1], [0, 0])

underlyingImage = testData[:,:,:,:].squeeze()

myDataNsless = getNoisyData(testData, 0, sysMtxRef)
myDataLRinvCR = getNoisyData(testData, 0, sysMtx)

stdScl = 0.41
stdVal = 10**(-pSNRval / 20) * stdScl# = std * (myDatNsless.shape[0:3]*)

# underlyingEpsilon = stdVal * (nbOfSingulars)**(1/2)
underlyingEpsilon = stdVal * (myDataNsless.shape[2])**(1/2)

myDataGen = getNoisyData(testData, stdVal, sysMtxRef)
print("Data without inverse crime SNR: ", 20 * torch.log10(torch.norm(myDataNsless) / torch.norm(myDataGen - myDataNsless)))
print("Data with inverse crime SNR: ", 20 * torch.log10(torch.norm(myDataLRinvCR) / torch.norm(myDataGen - myDataLRinvCR)))


datatC, lsqrInp = admmInputGenerator(myDataGen, U_, S_, V_, testData.shape)

Nbatch = datatC.shape[0]
sclScale = 1

imgSize = [n1, n2]

theWholeSize = ((-1, 1, *imgSize))

outImgs = list()
outDiags = list()
outMaxs = list()
outNetworkNorms = list()
outNetworkNrmses = list()
outPsnrs = list()
outHfens = list()
outL1Obj = list()
outTVobj = list()
outCritobj = list()

outX1 = list()
outX2 = list()

outName = list()
inPsnrs = list()
outNumels = list()

pSNRexp = 15 # 23

U, S, V = U_.clone(), S_.clone(), V_.clone()
simMtx2 = sysMtx

AtC = simMtx2.reshape(-1, n1*n2)
    

if underlyingEpsilon is not None:
    epsilonVal = underlyingEpsilon
else:
    epsilonVal = torch.norm(datatC, dim=1) * 10**(-pSNRexp/20)

if isinstance(epsilonVal, float) or epsilonVal.numel() == 1:
    epsilonVal = float(epsilonVal)

epsilon = epsilonVal
refVals = underlyingImage.reshape(Nbatch, -1)

print("Epsilon Value: ", epsilon)
print("Standard Deviation Value: ", epsilon / (datatC.numel())**(1/2))

datatC, lsqrInp = admmInputGenerator(myDataGen, U, S, V, theWholeSize)
datatC = datatC.reshape(Nbatch, -1)

if initializationTo == 0:
    x_in = torch.zeros_like(underlyingImage)
elif initializationTo == 1:
    x_in = lsqrInp.reshape(-1, n1, n2) #torch.linalg.lstsq(AtC, datatC.T).solution.T.reshape(-1, n1, n2)
else:
    x_in = F.linear(F.linear(datatC, AtC.T), torch.inverse(
        1 * torch.eye(n1 * n2).type_as(AtC.T) + AtC.T @ AtC)).reshape(-1, n1, n2)

inPsnrs.append(psnr(refVals, x_in.reshape(Nbatch, -1)))

updateStep = 1
verboseIn = 100
mu1 = 1
mu2 = 10
mu3 = 50
MaxIter = 200

theADMMclass = ADMMfncs(AtC, MaxIter, verboseIn, imgSize)
datatC = myDataGen.reshape(Nbatch, -1)


compMtx = AtC
Madmm = theADMMclass.MtC 

descriptors = descriptorsHere

for i, descriptor in enumerate(descriptors):
    
    if descriptor[:5] == 'ADMLD':
        testMode = 4 # ADMLD
    elif descriptor[:6] == 'DeqMPI':
        testMode = 5 # DeqMPI
    else:
        testMode = 0 # plug & play

    print(descriptor, "test Mode: ",testMode)
    
    if testMode == 0: # plug & play
        if not ("L1" in descriptor or "TV" in descriptor):

            model = getModel(descriptor[:-3])
            muScaleIter = 1
            mu2Scale = 1
            fnc1 = ppFnc2dnm(1/mu1, model, imgSize)

            fnc2 = softTV(1/30, imgSize, 10)
            fnc3 = softT(1/20)

            fncUse = None
            synCaller = False

            if 'tv' in descriptors[i]:
                fncUse = fnc2
            elif 'l1' in descriptors[i]:
                fncUse = fnc3
            elif 'lS' in descriptors[i]:
                fncUse = softTpos(1/20)
                synCaller = True

            if synCaller:
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj, x1, x2 = theADMMclass.ADMMreconDualSynthesis(
                    datatC, fnc1, fncUse, epsilon, refVals, mu2Scale=mu2Scale, muScaleIter=muScaleIter, x_in=x_in)
                outX1.append(x1)
                outX2.append(x2)
            else:
                if fncUse is None:
                    x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                        datatC, fnc1, fnc1, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)
                else:
                    x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.ADMMreconDual(
                        datatC, fnc1, fncUse, epsilon, refVals, mu2Scale=mu2Scale, muScaleIter=muScaleIter, x_in=x_in)
        else:
            muScaleIter = 70

            if not "TV" in descriptor:
                fnc3 = softT(1/(100 / 2))
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                    datatC, fnc3, fnc3, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)

            elif not "L1" in descriptor:
                fnc2 = softTV(1/(500 / 2), imgSize, 10)
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFnc(
                    datatC, fnc2, fnc2, 0, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)
            else:
                fnc3 = softTpos((1 - 0.1) / 10)
                fnc2 = softTV(0.1 / 10, imgSize, 10)
                x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.ADMMreconDual(
                    datatC, fnc2, fnc3, epsilon, refVals, mu2Scale=1, muScaleIter=muScaleIter, x_in=x_in)
    elif (testMode == 4) or (testMode == 5): # ADMM unrolled
        
        if testMode == 4:
            theMd, _ = getModelForADMMLD(descriptor)
        elif testMode == 5:
            numIter = 25
            theMd, _ = getModelForImplicitLD(descriptor, numIter)

        Vt = V_.T

        if datatC.shape[1] == U_.shape[1]:
            compData = datatC
        else:
            compData = F.linear(datatC.reshape(Nbatch, -1), U_.T)

        theWholeSize = ((-1, 1, *imgSize))
        x_rec = torch.zeros_like(refVals).reshape(theWholeSize)

        valInp = F.linear(compData / (S_ + 1e-4), V_).reshape(theWholeSize)

        batch_size_val = 256

        iii = 0
        compData = datatC
        
        if (testMode == 5): # run 
            Nimg = n1 * n2
            d0, d2 = torch.zeros_like(valInp).reshape(Nbatch, -1), torch.zeros_like(datatC)
            theIn = torch.cat((valInp.reshape(Nbatch, -1), d0.reshape(Nbatch, -1), d2.reshape(Nbatch, -1)), dim = 1 )
            while(iii < Nbatch - (Nbatch % batch_size_val)):
                imgSize2 = (batch_size_val, *theWholeSize[1:])
                theFixedPts = (compData[iii:iii+batch_size_val], compMtx, Madmm, epsilonVal, imgSize2)
                x_rec[iii:iii+batch_size_val] = theMd(theIn[iii:iii+batch_size_val], theFixedPts)[:, :Nimg].reshape(imgSize2)
                iii += batch_size_val
            imgSize2 = (compData[iii:].shape[0], *theWholeSize[1:])
            theFixedPts = (compData[iii:], compMtx, Madmm, epsilonVal, imgSize2)
            x_rec[iii:] = theMd(theIn[iii:], theFixedPts)[:, :Nimg].reshape(imgSize2)
        elif (testMode == 4):
            while(iii < valInp.shape[0]-(valInp.shape[0] % batch_size_val)):
                x_rec[iii:iii+batch_size_val, :, :, :] = theMd(compData[iii:iii+batch_size_val], compMtx, 1, valInp[iii:iii+batch_size_val], Madmm, epsilonVal)
                iii += batch_size_val
            x_rec[iii:] = theMd(compData[iii:], compMtx, 1, valInp[iii:], Madmm, epsilonVal)

        x_rec = x_rec.reshape(Nbatch, -1)

    if testMode > 0:
        outDiag, outMax, outPsnr, outHfen, l1Obj, TVobj = theADMMclass.calculateDiagnoseVals(x_rec, refVals)
        outDiag /= float(torch.norm(refVals))
        outHfen /= theADMMclass.hfenFnc(refVals)
        critObj = torch.sqrt(torch.sum((F.linear(x_rec, AtC) - datatC).reshape(Nbatch, -1).abs() ** 2, dim = 1)).detach().cpu().numpy() / epsilon

    outImgs.append(x_rec)
    outDiags.append(outDiag)
    outMaxs.append(outMax)
    outPsnrs.append(outPsnr)
    outHfens.append(outHfen)
    outL1Obj.append(l1Obj)
    outTVobj.append(TVobj)
    outCritobj.append(critObj)

    if testMode == 0:
        outNetworkNorms.append(inpOutNorms)
        outNetworkNrmses.append(inpOutNrmses)

for i, descriptor in enumerate(descriptorsLD):
    
    if descriptor[:5] == 'ADMLD':
        testMode = 4 # ADMLD
    elif descriptor[:6] == 'DeqMPI':
        testMode = 5 # Implicit
    
    print(descriptor, "test Mode: ",testMode)
    
    if (testMode == 4) or (testMode == 5): # ADMM unrolled
        
        if testMode == 4:
            theMd, _ = getModelForADMMLD(descriptor)
        elif testMode == 5:
            numIter = 25
            theMd2, _ = getModelForImplicitLD(descriptor, numIter)
            theMd = theMd2.f

        fnc1 = ppFnc2dnm(1, theMd.sharedNet, imgShape = imgSize)
        fncLD = theMd.consistencyNet
        x_rec, outDiag, outPsnr, outHfen, inpOutNorms, inpOutNrmses, outMax, l1Obj, TVobj, critObj = theADMMclass.afterFncLD(
            datatC, fnc1, fnc1, 0, epsilon, refVals, mu2Scale=1, muScaleIter=1, consistencyFnc = fncLD, x_in=x_in)

        x_rec = x_rec.reshape(Nbatch, -1)
    
    outImgs.append(x_rec)
    outDiags.append(outDiag)
    outMaxs.append(outMax)
    outPsnrs.append(outPsnr)
    outHfens.append(outHfen)
    outL1Obj.append(l1Obj)
    outTVobj.append(TVobj)
    outCritobj.append(critObj)
    if testMode == 0:
        outNetworkNorms.append(inpOutNorms)
        outNetworkNrmses.append(inpOutNrmses)

if pSNRval < 10:
    lambdaDeger = 10
elif pSNRval < 30:
    lambdaDeger = 1
elif pSNRval < 40:
    lambdaDeger = 0.1
else:
    lambdaDeger = 0.01

maxIt = 10
xART, diagOutVals = ART(sysMtx, myDataGen.reshape(myDataGen.shape[0], -1), maxIter = maxIt, lambdaVal = lambdaDeger, order = None, energy = None, underlyingImage = underlyingImage)


def selFnc(x):
    return x[0, 1:]

def selFnc(x):
    if len(x.shape) > 1:
        return x[:, 1:].mean(0)
    elif len(x.shape) == 1:
        return x.mean(0).repeat(MaxIter - 1)
    else:
        return x.repeat(MaxIter - 1)

namesOrdered = copy.deepcopy([*descriptors, *descriptorsLD])

for d, e, epsVls in zip(outPsnrs, namesOrdered, outCritobj):
    dSel = selFnc(d)
    dFin = dSel[-1]
    dMax = dSel.max()
    epsVl = round(selFnc(epsVls)[-1], 2)
    print('Final pSNR: ', round(dFin, 2), ' max pSNR: ', round(dMax, 2), ', crit: ', epsVl ,', method: {0:60}'.format(e))


preProcessValues = lambda x: x.reshape(1, 3730, -1).repeat( 1 + 199 * (2 - len(x.shape)) , 2)

saveImgs = np.concatenate([indivImg.unsqueeze(0).cpu().numpy() for indivImg in outImgs])
saveDiags = np.concatenate([preProcessValues(indivImg) for indivImg in outDiags])
savePsnrs = np.concatenate([preProcessValues(indivImg) for indivImg in outPsnrs])
saveHfens = np.concatenate([preProcessValues(indivImg) for indivImg in outHfens])
saveL1Obj = np.concatenate([preProcessValues(indivImg) for indivImg in outL1Obj])
saveTVobj = np.concatenate([preProcessValues(indivImg) for indivImg in outTVobj])
saveCritobj = np.concatenate([preProcessValues(indivImg) for indivImg in outCritobj])
saveNetworkNorms = np.concatenate([preProcessValues(indivImg) for indivImg in outNetworkNorms])
saveNetworkNrmses = np.concatenate([preProcessValues(indivImg) for indivImg in outNetworkNrmses])
# saveMaxs = np.concatenate([preProcessValues(indivImg) for indivImg in outMaxs])

savemat('Simulated{0}dBresults.mat'.format(pSNRval), {'descriptors':[*descriptors, *descriptorsLD], 'saveImgs': saveImgs, 'saveDiags': saveDiags.astype(np.float32), 'savePsnrs': savePsnrs.astype(np.float32), 'saveHfens': saveHfens.astype(np.float32), 
    'saveL1Obj': saveL1Obj.astype(np.float32), 'saveTVobj': saveTVobj.astype(np.float32), 'saveCritobj': saveCritobj.astype(np.float32), 'saveNetworkNorms': saveNetworkNorms, 'saveNetworkNrmses': saveNetworkNrmses, 'xART': xART.cpu().detach().numpy(), 'ARTdiags': diagOutVals})
savemat('Simulated{0}dBresultsOnlyPsnr.mat'.format(pSNRval), {'descriptors':[*descriptors, *descriptorsLD], 'savePsnrs': savePsnrs.astype(np.float32), 'ARTdiags': diagOutVals})
savemat('Simulated{0}dBdata.mat'.format(pSNRval), {'myDataGen': myDataGen.cpu().detach().numpy(), 'testData': testData.cpu().detach().numpy()})

