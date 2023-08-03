# from cv2 import UMAT_SUBMATRIX_FLAG
from scipy.io import savemat, loadmat
import numpy as np
from torch import nn
import torch.nn.functional as F
from scipy.io import loadmat, savemat
import time

from torch.utils.data import Dataset, DataLoader
import wandb
import torch

def transformDataset(data, imgSizes, rescaleVals, randVals = None):

    dims = len(imgSizes)

    reScaleMin, reScaleMax = rescaleVals
    imgSize = data.shape

    if dims == 2:
        n1, n2 = imgSizes

        data -= data.reshape(imgSize[0], imgSize[1], -1).min(dim = 2).values[:,:,None,None]
        data /= data.reshape(imgSize[0], imgSize[1], -1).max(dim = 2).values[:,:,None,None]

        if (n1 < data.shape[2]) or (n2 < data.shape[3]):
            if randVals is None:
                rand1 = torch.randint(low = 0, high = data.shape[2] - n1, size = (1,))
                rand2 = torch.randint(low = 0, high = data.shape[3] - n2, size = (1,))
            else:
                rand1 = randVals[0]
                rand2 = randVals[1]
        else:
            rand1 = 0
            rand2 = 0
        data = data[:, :, rand1:rand1 + n1, rand2:rand2 + n2]
    else:
        n1, n2, n3 = imgSizes # n1 is 16

        data -= data.reshape(imgSize[0], imgSize[1], -1).min(dim = 2).values[:,:,None,None,None]
        data /= data.reshape(imgSize[0], imgSize[1], -1).max(dim = 2).values[:,:,None,None,None]

        diffS = np.array(imgSizes) - np.array(data.shape[2:])
        diffS *= (diffS > 0)
        data = F.pad(data, (diffS[2], diffS[2], diffS[1], diffS[1], diffS[0], diffS[0])) # new data size is 

        if (n1 < data.shape[2]) or (n2 < data.shape[3]) or (n3 < data.shape[3]):
            if randVals is None:
                rand1 = torch.randint(low = 0, high = data.shape[2] - n1, size = (1,))
                rand2 = torch.randint(low = 0, high = data.shape[3] - n2, size = (1,))
                rand3 = torch.randint(low = 0, high = data.shape[4] - n3, size = (1,))
            else:
                rand1 = randVals[0]
                rand2 = randVals[1]
                rand3 = randVals[2]
        else:
            rand1 = 0
            rand2 = 0
            rand3 = 0

        data = data[:, :, rand1:rand1 + n1, rand2:rand2 + n2, rand3:rand3 + n3]
    
    if reScaleMax > 0:
        randScale = torch.rand((imgSize[0], 1, 1, 1), device = data.device) * reScaleMax + reScaleMin
        if dims == 3:
            randScale = randScale.reshape(imgSize[0], 1, 1, 1, 1)
    else:
        randScale = 1

    data *= randScale
    return data

def getNoisyData(data, stdVal, sysMtx, colored = 0):
    genData = F.linear(data.reshape(data.shape[0], 1, -1), sysMtx)
    if colored > 0:
        if colored > 2: # laplacian noise
            r1 = -torch.log(torch.rand_like(genData)) + torch.log(torch.rand_like(genData))

            genNoise = (stdVal / (2)**(1/2)) * r1
            # genNoise = stdVal * theScale * ( (1 - colored) + colored * torch.rand_like(genData)) * torch.randn_like(genData) # generate random
        else:
            # colored with max 1
            # theScale = ( 3 * colored / (  1 - (1 - colored)**3 ))**(1/2)
            # genNoise = stdVal * theScale * ( (1 - colored) + colored * torch.rand_like(genData)) * torch.randn_like(genData) # generate random
            # colored with mean 1
            theScale = ( 3 * colored / (  (1+colored/2)**3 - (1 - colored/2)**3 ))**(1/2)
            genNoise = stdVal * theScale * ( (1 - colored) + colored * torch.rand_like(genData)) * torch.randn_like(genData) # generate random
    else:
        genNoise = stdVal * torch.randn_like(genData) # generate random
    genData += genNoise # add noise in original data domain
    return genData

def admmInputGenerator(genData, U_, S_, V_, imgSize):

    compData = F.linear(genData, U_.T) # compressed data

    lsqrInp = F.linear(compData / (S_ + 1e-4), V_).reshape(imgSize)

    return compData, lsqrInp


 
def trainADMMandE2EandImplicit(model, epoch_nb, loss, optimizer, scheduler, trainDataset, valDataset, batch_size_train, batch_size_val, theSysl, sysMtxl, Ul, Sl, Vl, Ml, sysMtxlHR = None, invCrimeFnc = None, imgSizes = [32, 32] \
    , rescaleVals = [1, 1], saveModelEpoch=0, valEpoch=0, saveDirectory='', pSNRval=30, wandbFlag=False, fixedNoiseStdFlag=False, nbOfSingulars=0, lambdaVal = 1, mode = 0, coloredNoise = 0):
    
    variablesEpsFlag = pSNRval == 0
    stdVal = 0.41 * 10**(-(pSNRval) / 20)

    N = np.prod(imgSizes)
    # epsilonVal = stdVal * (N)**(1/2) * lambdaVal
    if sysMtxlHR is None:
        epsOffset = 0
    else:
        epsOffset = 0.087 * 0
    epsilonVal = ((stdVal * (N)**(1/2) * lambdaVal)**2 + epsOffset)**(1/2)

    trainLosses = torch.zeros(epoch_nb)
    trainNrmses = torch.zeros(epoch_nb)
    trainPsnrs = torch.zeros(epoch_nb)
    valLosses = list()
    valNrmses = list()
    valPsnrs = list()
    trainLoader = DataLoader(trainDataset, batch_size_train, shuffle=True)
    valLoader = DataLoader(valDataset, valDataset.__len__(), shuffle=False)
    maxPsnrVal = -1000
    for epoch in range(int(epoch_nb)):
        wandbLoggerDict = {'epoch': epoch}
        tempLosses = list()
        model.train()
        tempNrmseNumeratorSquare = 0
        tempNrmseDenumeratorSquare = 0
        tempNumel = 0
        tempTime = time.time()

        if saveModelEpoch > 0:
            if (epoch + 1 % saveModelEpoch == 0):
                torch.save(model.state_dict(), saveDirectory+r"/" +
                           "epoch" + str(epoch + 1) + ".pth")

        for idx, dataHR in enumerate(trainLoader, 0):

            myRndInd = int(torch.randint(low = 0, high = len(Ul), size = (1,)))

            dataHR = dataHR.cuda().float()
            # compressedRep = 1, noisySys
            if invCrimeFnc is None:
                data = dataHR
                data = transformDataset(data, imgSizes, rescaleVals)
            else:
                dataHR = transformDataset(dataHR, [b*2 for b in imgSizes], rescaleVals)
                data = invCrimeFnc(dataHR)

            if variablesEpsFlag:
                stdVal = 0.41 * 10**(-(float(torch.randint(low = 20, high = 40, size = (1,)))) / 20)
                epsilonVal = ((stdVal * (N)**(1/2) * lambdaVal)**2 + epsOffset)**(1/2)

            if sysMtxlHR is None:
                theNoisyData = getNoisyData(data, stdVal, sysMtxl[myRndInd], coloredNoise)
            else:
                theNoisyData = getNoisyData(dataHR, stdVal, sysMtxlHR[myRndInd], coloredNoise)

            compData, lsqrInp = admmInputGenerator(theNoisyData, Ul[myRndInd], Sl[myRndInd], Vl[myRndInd], data.shape)

            epsilonVal = ((stdVal * (sysMtxl[myRndInd].shape[0])**(1/2) * lambdaVal)**2 + epsOffset)**(1/2)

            if (mode == 1) or (mode == 3): # ADMM
                modelOut = model(datatC = theNoisyData, AtC = sysMtxl[myRndInd], x_in = lsqrInp, MtC = Ml[myRndInd], epsilon = epsilonVal)
            elif mode == 2: # ADMM Implicit
                imgSize = data.shape
                Nimg = N
                AlsqrInp = F.linear(lsqrInp.reshape(imgSize[0], -1), sysMtxl[myRndInd])
                d0, d2 = torch.zeros_like(lsqrInp).reshape(imgSize[0], -1), torch.zeros_like(AlsqrInp)
                theIn = torch.cat((lsqrInp.reshape(imgSize[0], -1), d0.reshape(imgSize[0], -1), d2.reshape(imgSize[0], -1)), dim = 1 )
                theFixedPts = (theNoisyData, sysMtxl[myRndInd], Ml[myRndInd], epsilonVal, imgSize)
                modelOut = model(theIn, theFixedPts)[:, :Nimg].reshape(imgSize)

            model.zero_grad()
            model_loss = loss(modelOut, data)
            model_loss.backward()
            optimizer.step()
                    
            with torch.no_grad():
                if len(data.shape) == 5:
                    percent = 10
                    if idx%int(trainDataset.__len__()/batch_size_train/percent)==0:
                        print('Epoch {0:d} | {1:d}% | batch nrmse: {2:.5f}'.format(epoch,percent*idx//int(trainDataset.__len__()/batch_size_train/percent),(float(torch.norm(modelOut-data)))/(float(torch.norm(data))))) # myflag
                tempLosses.append(float(model_loss))
                tempNrmseNumeratorSquare += (
                    float(torch.norm(modelOut-data)))**2
                tempNrmseDenumeratorSquare += (float(torch.norm(data)))**2
                tempNumel += modelOut.numel()
        # back to epoch
        model.eval()
        scheduler.step()

        trainLosses[epoch] = sum(tempLosses)/len(tempLosses)
        trainNrmses[epoch] = (tempNrmseNumeratorSquare /
                                tempNrmseDenumeratorSquare)**(1/2)
        trainPsnrs[epoch] = 20 * \
            torch.log10(1 / (tempNrmseDenumeratorSquare**(1/2) *  # Should we correct 1 -> valGround.max()
                             trainNrmses[epoch] / (tempNumel) ** (1/2)))
        epochTime = time.time() - tempTime
        if wandbFlag:
            wandbLoggerDict["train_loss"] = trainLosses[epoch]
            wandbLoggerDict["train_nrmse"] = trainNrmses[epoch]
            wandbLoggerDict["train_psnr"] = trainPsnrs[epoch]

        print("Epoch: {0}, Train Loss = {1:.6f}, Train nRMSE = {2:.6f}, Train pSNR = {3:.6f}, time elapsed = {4:.6f}".format(epoch,
                                                                                                                             trainLosses[epoch], trainNrmses[epoch], trainPsnrs[epoch], epochTime))

        if valEpoch > 0:
            if epoch % valEpoch == 0:
                with torch.no_grad():

                    model.eval()            

                    myRndInd = int(torch.randint(low = 0, high = len(Ul), size = (1,)))

                    valInpHR = next(iter(valLoader)).cuda().float()


                    if invCrimeFnc is None:
                        valInp = valInpHR
                        valInp = transformDataset(valInp, imgSizes, rescaleVals)
                    else:
                        valInpHR = transformDataset(valInpHR, [b*2 for b in imgSizes], rescaleVals)
                        valInp = invCrimeFnc(valInpHR)

                    if variablesEpsFlag:
                        stdVal = 0.41 * 10**(-(float(torch.randint(low = 20, high = 40, size = (1,)))) / 20)
                        epsilonVal = ((stdVal * (N)**(1/2) * lambdaVal)**2 + epsOffset)**(1/2)
                        
                    if sysMtxlHR is None:
                        theNoisyData = getNoisyData(valInp, stdVal, sysMtxl[myRndInd], coloredNoise)
                    else:
                        theNoisyData = getNoisyData(valInpHR, stdVal, sysMtxlHR[myRndInd], coloredNoise)

                    # valInp = transformDataset(valInp, imgSizes, rescaleVals)

                    valGround = valInp.clone()
                    valOut = torch.zeros_like(valInp)

                    # theNoisyData = getNoisyData(valGround, stdVal, sysMtxl[myRndInd])
                    compData, valInp = admmInputGenerator(theNoisyData, Ul[myRndInd], Sl[myRndInd], Vl[myRndInd], valGround.shape)

                    iii = 0
                    epsilonVal = ((stdVal * (sysMtxl[myRndInd].shape[0])**(1/2) * lambdaVal)**2 + epsOffset)**(1/2)
                    # stdVal * (sysMtxl[myRndInd].shape[0])**(1/2) * lambdaVal
                    if (mode == 1) or (mode == 3): # ADMM
                        while(iii < valInp.shape[0]-(valInp.shape[0] % batch_size_val)):
                            valOut[iii:iii+batch_size_val] = model(datatC = theNoisyData[iii:iii+batch_size_val], AtC = sysMtxl[myRndInd], x_in = valInp[iii:iii+batch_size_val], MtC = Ml[myRndInd], epsilon = epsilonVal)
                            iii += batch_size_val
                        valOut[iii:] = model(datatC = theNoisyData[iii:], AtC = sysMtxl[myRndInd], x_in = valInp[iii:], MtC = Ml[myRndInd], epsilon = epsilonVal)
                    elif mode == 2: # ADMM Implicit
                        imgSize = valInp.shape
                        Nimg = N
                        AvalInp = F.linear(valInp.reshape(imgSize[0], -1), sysMtxl[myRndInd])
                        d0, d2 = torch.zeros_like(valInp).reshape(imgSize[0], -1), torch.zeros_like(AvalInp)
                        theIn = torch.cat((valInp.reshape(imgSize[0], -1), d0.reshape(imgSize[0], -1), d2.reshape(imgSize[0], -1)), dim = 1 )
                        while(iii < valInp.shape[0]-(valInp.shape[0] % batch_size_val)):
                            imgSize2 = (batch_size_val, *imgSize[1:])
                            theFixedPts = (theNoisyData[iii:iii+batch_size_val], sysMtxl[myRndInd], Ml[myRndInd], epsilonVal, imgSize2)
                            valOut[iii:iii+batch_size_val] = model(theIn[iii:iii+batch_size_val], theFixedPts)[:, :Nimg].reshape(imgSize2)
                            iii += batch_size_val
                        imgSize2 = (theNoisyData[iii:].shape[0], *imgSize[1:])
                        theFixedPts = (theNoisyData[iii:], sysMtxl[myRndInd], Ml[myRndInd], epsilonVal, imgSize2)
                        valOut[iii:] = model(theIn[iii:], theFixedPts)[:, :Nimg].reshape(imgSize2)

                    valLoss = float(nn.L1Loss()(valGround, valOut))
                    valNrmse = float(torch.norm(
                        valGround-valOut)/torch.norm(valGround))
                    valPSNR = float(20 *
                                    torch.log10(1 / (torch.norm(valGround) *  # Should we correct 1 -> valGround.max()
                                                     valNrmse / (valOut.numel()) ** (1/2))))

                    valPSNR_avg = (20 *
                                    torch.log10(1 / (torch.norm((valGround-valOut).reshape(valGround.shape[0], -1), dim = (1)).squeeze() / (valOut[0,0].numel()) ** (1/2))))

                    valLosses.append(valLoss)
                    valNrmses.append(valNrmse)
                    valPsnrs.append(valPSNR)
                if wandbFlag:
                    currentAvg = valPSNR_avg.mean(0)
                    if maxPsnrVal < currentAvg:
                        maxPsnrVal = currentAvg
                        torch.save(model.state_dict(), saveDirectory+r"/" +
                                "epoch" + str(epoch) + "max.pth")
                        
                    wandbLoggerDict["valid_nRMSE"] = valNrmse
                    wandbLoggerDict["ref_nRMSE"] = torch.norm(valInp-valGround)/torch.norm(valGround)
                    wandbLoggerDict["valid_pSNR"] = valPSNR
                    wandbLoggerDict["valid_pSNRavg"] = valPSNR_avg.mean(0)
                    wandbLoggerDict["valid_pSNRstd"] = valPSNR_avg.std(0)
                    wandbLoggerDict["valid_maxPSNR"] = maxPsnrVal
                    wandbLoggerDict["valid_loss"] = valLoss
                print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(
                    epoch, valLoss, valNrmse, valPSNR))
        if wandbFlag:
            wandb.log(wandbLoggerDict)
    
    torch.save(model.state_dict(), saveDirectory+r"/" +
               "epoch" + str(epoch + 1) + "END.pth")
    return model, [trainLosses.numpy(), trainNrmses.numpy(), trainPsnrs.numpy()], [np.array(valLosses), np.array(valNrmses), np.array(valPsnrs)]



def returnNsTerm(fixedNoiseStdFlag = False,shape=0, maxNoiseStd=0.05, minNoiseStd=0, useDev = torch.device("cpu")):
    if fixedNoiseStdFlag:
        noiseStd = maxNoiseStd
        return torch.randn(shape, device = useDev) * noiseStd
    else:
        noiseStd = (torch.rand(shape, device = useDev)*(maxNoiseStd-minNoiseStd)+minNoiseStd)
        return torch.randn(shape, device = useDev) * noiseStd




def trainDenoiser(model, epoch_nb, loss, optimizer, scheduler, trainDataset, valDataset, batch_size_train, batch_size_val, rescaleVals = [1, 1], saveModelEpoch=0, valEpoch=0, saveDirectory='', maxNoiseStd = 0.1, optionalMessage="", wandbFlag=False, fixedNoiseStdFlag = False, minNoiseStd =0, dims = 2):
    trainLosses = torch.zeros(epoch_nb)
    trainNrmses = torch.zeros(epoch_nb)
    trainPsnrs = torch.zeros(epoch_nb)
    valLosses = list()
    valNrmses = list()
    valPsnrs = list()
    trainLoader = DataLoader(trainDataset, batch_size_train, shuffle=True)
    valLoader = DataLoader(valDataset, valDataset.__len__(), shuffle=False)
    reScaleMin, reScaleMax = rescaleVals

    for epoch in range(1,1+int(epoch_nb)):
        tempLosses = list()
        model.train()
        tempNrmseNumeratorSquare = 0
        tempNrmseDenumeratorSquare = 0
        tempNumel = 0
        tempTime = time.time()

        if saveModelEpoch > 0:
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), saveDirectory+r"/"+ optionalMessage +"epoch"+ str(epoch)+ ".pth")

        for idx, data in enumerate(trainLoader, 0):
            data = data.float().cuda()

            data = transformDataset(data, [*data.shape[2:]], rescaleVals)
            
            noiseTerm = returnNsTerm(fixedNoiseStdFlag = fixedNoiseStdFlag, shape=data.shape,\
                                     maxNoiseStd=maxNoiseStd, minNoiseStd=minNoiseStd, useDev = data.device)
            noisyInp = data + noiseTerm
            modelOut = model(noisyInp)
            model.zero_grad()
            model_loss = loss(modelOut, data)
            model_loss.backward()
            optimizer.step()
            if dims == 3:
                percent = 10
                if idx%int(trainDataset.__len__()/batch_size_train/percent)==0:
                    print('Epoch {0:d} | {1:d}% | batch nrmse: {2:.5f}'.format(epoch,percent*idx//int(trainDataset.__len__()/batch_size_train/percent),(float(torch.norm(modelOut-data)))/(float(torch.norm(data))))) # myflag

            with torch.no_grad():
                tempLosses.append(float(model_loss))
                tempNrmseNumeratorSquare += (float(torch.norm(modelOut-data)))**2
                tempNrmseDenumeratorSquare += (float(torch.norm(data)))**2
                tempNumel += modelOut.numel()
        # back to epoch
        model.eval()
        scheduler.step()
            
        trainLosses[epoch-1] = sum(tempLosses)/len(tempLosses)
        trainNrmses[epoch-1] = (tempNrmseNumeratorSquare/tempNrmseDenumeratorSquare)**(1/2)
        trainPsnrs[epoch-1] = 20 * \
                    torch.log10(1 / (tempNrmseDenumeratorSquare**(1/2) * #Should we correct 1 -> valGround.max()
                                trainNrmses[epoch-1] / (tempNumel) ** (1/2)))
        epochTime = time.time() - tempTime
        if wandbFlag:
            wandb.log({"train_loss": trainLosses[epoch-1], "train_nrmse": trainNrmses[epoch-1], "train_psnr": trainPsnrs[epoch-1]})
        
        print("Epoch: {0}, Train Loss = {1:.6f}, Train nRMSE = {2:.6f}, Train pSNR = {3:.6f}, time elapsed = {4:.6f}".format(epoch, 
                                                trainLosses[epoch-1], trainNrmses[epoch-1], trainPsnrs[epoch-1], epochTime))        
        
        if valEpoch>0:
            if epoch % valEpoch == 0:
                with torch.no_grad():
                    model.eval()
                    valInp = next(iter(valLoader))

                    valGround = valInp.clone()
                    valGround = transformDataset(valGround, [*valInp.shape[2:]], rescaleVals)
                    noiseTerm = returnNsTerm(fixedNoiseStdFlag = fixedNoiseStdFlag,shape=valInp.shape, \
                                     maxNoiseStd=maxNoiseStd, minNoiseStd=minNoiseStd, useDev = valGround.device)
                    valInpVal = valGround + noiseTerm

                    valOut = torch.zeros_like(valGround)
                    deviceVal = valGround.device #'cuda' if valOut.is_cuda else 'cpu'
                    
                    iii = 0
                    while(iii < valInpVal.shape[0]-(valInpVal.shape[0] % batch_size_val)):
                        valInpC = valInpVal[iii:iii+batch_size_val].float().cuda()
                        valOut[iii:iii+batch_size_val] = model(valInpC).to(deviceVal)
                        iii += batch_size_val
                    
                    valInpC = valInpVal[iii:].float().cuda()
                    valOut[iii:] = model(valInpC).to(deviceVal)
                
                    valLoss = float(nn.L1Loss()(valGround, valOut))
                    valNrmse = float(torch.norm(valGround-valOut)/torch.norm(valGround))
                    valPSNR= float(20 * \
                    torch.log10(1 / (torch.norm(valGround) * #Should we correct 1 -> valGround.max()
                                valNrmse / (valOut.numel()) ** (1/2))))
                    # valPSNR_avg = (20 *
                    #                 torch.log10(1 / (torch.norm(valGround-valOut, dim = (2, 3)).squeeze() / (valOut[0,0].numel()) ** (1/2))))
                    valPSNR_avg = (20 *
                                    torch.log10(1 / (torch.norm(valGround.reshape(valGround.shape[0],-1)-valOut.reshape(valGround.shape[0],-1), dim = (1)).squeeze() / (valOut[0,0].numel()) ** (1/2))))

                    valLosses.append(valLoss)
                    valNrmses.append(valNrmse)
                    valPsnrs.append(valPSNR)
                if wandbFlag:
                    wandb.log({"valid_nRMSE": valNrmse,
                               "ref_nRMSE": torch.norm(valInp-valGround)/torch.norm(valGround),
                              'valid_pSNR': valPSNR,
                              'valid_pSNRavg': valPSNR_avg.mean(0),
                              'valid_pSNRstd': valPSNR_avg.std(0),
                               'valid_loss': valLoss})
                print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(epoch, valLoss, valNrmse, valPSNR))

    if wandbFlag:
        wandb.log({"valid_nRMSE": valNrmse,
                  'valid_pSNR': valPSNR,
                  'valid_pSNRavg': valPSNR_avg.mean(0),
                  'valid_pSNRstd': valPSNR_avg.std(0),
                   'valid_loss': valLoss})
    print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(epoch, valLoss, valNrmse, valPSNR))        
    torch.save(model.state_dict(), saveDirectory+r"/"+ optionalMessage +"epoch"+ str(epoch)+ "END.pth")
    return model, [trainLosses.numpy(), trainNrmses.numpy(), trainPsnrs.numpy()], [np.array(valLosses), np.array(valNrmses), np.array(valPsnrs)]

def getDCNoisyData(data, stdVal, sysMtx):
    noiselessData = F.linear(data.reshape(data.shape[0], 1, -1), sysMtx)
    genNoise = stdVal * torch.randn_like(noiselessData) # generate random
    genData = noiselessData + genNoise # add noise in original data domain
    return noiselessData, genData

from reconUtils import proj2Tmtx

def trainDCdenoiserPsi(model, epoch_nb, loss, optimizer, scheduler, trainDataset, valDataset, batch_size_train, batch_size_val, theSysl, sysMtxl, Ul, Sl, Vl, noisySys = 0, imgSizes = [32, 32] \
    , rescaleVals = [1, 1], saveModelEpoch=0, valEpoch=0, saveDirectory='', pSNRval=40, wandbFlag=False, fixedNoiseStdFlag=False, nbOfSingulars=0, lambdaVal = 100):
    
    trainingNs = 0.41 * 10**(-pSNRval / 20)
    trainLosses = torch.zeros(epoch_nb)
    trainNrmses = torch.zeros(epoch_nb)
    trainPsnrs = torch.zeros(epoch_nb)
    valLosses = list()
    valNrmses = list()
    valPsnrs = list()
    trainLoader = DataLoader(trainDataset, batch_size_train, shuffle=True)
    valLoader = DataLoader(valDataset, valDataset.__len__(), shuffle=False)
    for epoch in range(int(epoch_nb)):
        wandbLoggerDict = {'epoch': epoch}
        tempLosses = list()
        model.train()
        tempNrmseNumeratorSquare = 0
        tempRefNrmseNumeratorSquare = 0
        tempNrmseDenumeratorSquare = 0
        tempNumel = 0
        tempTime = time.time()

        if saveModelEpoch > 0:
            if (epoch + 1 % saveModelEpoch == 0):
                torch.save(model.state_dict(), saveDirectory+r"/" +
                           "epoch" + str(epoch + 1) + ".pth")

        for idx, data in enumerate(trainLoader, 0):

            myRndInd = int(torch.randint(low = 0, high = len(sysMtxl), size = (1,)))

            data = data.float().cuda()
            data = transformDataset(data, imgSizes, rescaleVals)
                # compressedRep = 1, noisySys
            noiselessData, theNoisyData = getDCNoisyData(data, noisySys, sysMtxl[myRndInd])
            # trainingAddNoise = theNoisyData + trainingNs * torch.randn_like(theNoisyData)
            trainingAddNoise = noiselessData + trainingNs * torch.randn_like(theNoisyData)

            Nbatch = noiselessData.shape[0]

            in1 = trainingAddNoise.reshape(Nbatch, 1, -1)
            in2 = theNoisyData.reshape(Nbatch, 1, -1)
            noiselessData = noiselessData.reshape(Nbatch, 1, -1)

            epsilonVal = noisySys * (sysMtxl[myRndInd].shape[0])**(1/2) * lambdaVal
            modelExpected = proj2Tmtx(in1, in2, epsilonVal).reshape(Nbatch, 1, -1)

            modelOut = model(in1,
                        in2, epsilonVal, Ua = 1).reshape(Nbatch, 1, -1)

            model.zero_grad()
            model_loss = loss(modelOut, modelExpected)
            model_loss.backward()
            optimizer.step()

                    
            with torch.no_grad():
                if len(data.shape) == 5:
                    percent = 10
                    if idx%int(trainDataset.__len__()/batch_size_train/percent)==0:
                        print('Epoch {0:d} | {1:d}% | batch nrmse: {2:.5f}'.format(epoch,percent*idx//int(trainDataset.__len__()/batch_size_train/percent),(float(torch.norm(modelOut-data)))/(float(torch.norm(data))))) # myflag
                tempLosses.append(float(model_loss))
                tempNrmseNumeratorSquare += (
                    float(torch.norm(modelOut-modelExpected)))**2
                tempRefNrmseNumeratorSquare += (
                    float(torch.norm(theNoisyData-modelExpected)))**2
                tempNrmseDenumeratorSquare += (float(torch.norm(modelExpected)))**2
                tempNumel += modelOut.numel()
        # back to epoch
        model.eval()
        scheduler.step()

        trainLosses[epoch] = sum(tempLosses)/len(tempLosses)
        trainNrmses[epoch] = (tempNrmseNumeratorSquare /
                                tempNrmseDenumeratorSquare)**(1/2)
        trainPsnrs[epoch] = 20 * \
            torch.log10(1 / (tempNrmseDenumeratorSquare**(1/2) *  # Should we correct 1 -> valGround.max()
                             trainNrmses[epoch] / (tempNumel) ** (1/2)))
        epochTime = time.time() - tempTime
        if wandbFlag:
            wandbLoggerDict["train_loss"] = trainLosses[epoch]
            wandbLoggerDict["train_nrmse"] = trainNrmses[epoch]
            wandbLoggerDict["train_refnrmse"] = (tempRefNrmseNumeratorSquare / tempNrmseDenumeratorSquare)**(1/2)
            wandbLoggerDict["train_psnr"] = trainPsnrs[epoch]

        print("Epoch: {0}, Train Loss = {1:.6f}, Train nRMSE = {2:.6f}, Train pSNR = {3:.6f}, time elapsed = {4:.6f}".format(epoch,
                                                                                                                             trainLosses[epoch], trainNrmses[epoch], trainPsnrs[epoch], epochTime))

        if valEpoch > 0:
            if epoch % valEpoch == 0:
                with torch.no_grad():

                    model.eval()            

                    myRndInd = int(torch.randint(low = 0, high = len(sysMtxl), size = (1,)))

                    valInp = next(iter(valLoader)).cuda().float()
                    valInp = transformDataset(valInp, imgSizes, rescaleVals)

                    # noiselessData, theNoisyData = getDCNoisyData(data, noisySys, sysMtxl[myRndInd])

                    noiselessData, theNoisyData = getDCNoisyData(valInp, noisySys, sysMtxl[myRndInd])
                    Nbatch = noiselessData.shape[0]

                    noiselessData = noiselessData.reshape(Nbatch, 1, -1)
                    theNoisyData = theNoisyData.reshape(Nbatch, 1, -1)
                    trainingAddNoise = noiselessData + trainingNs * torch.randn_like(theNoisyData)
                    # trainingAddNoise = theNoisyData + trainingNs * torch.randn_like(theNoisyData)

                    valOut = torch.zeros_like(noiselessData)

                    iii = 0

                    epsilonVal = noisySys * (sysMtxl[myRndInd].shape[0])**(1/2) * lambdaVal

                    in1 = trainingAddNoise
                    in2 = theNoisyData

                    modelExpected = proj2Tmtx(in1, in2, epsilonVal).reshape(Nbatch, 1, -1)

                    while(iii < valInp.shape[0]-(valInp.shape[0] % batch_size_val)):
                        valOut[iii:iii+batch_size_val] = model(in1[iii:iii+batch_size_val], in2[iii:iii+batch_size_val], \
                            epsilonVal, Ua = 1)
                        iii += batch_size_val
                    valOut[iii:] = model(in1[iii:], in2[iii:], epsilonVal, Ua = 1)

                    valLoss = float(nn.L1Loss()(modelExpected, valOut))
                    valNrmse = float(torch.norm(
                        modelExpected-valOut)/torch.norm(modelExpected))
                    valPSNR = float(20 *
                                    torch.log10(1 / (torch.norm(modelExpected) *  # Should we correct 1 -> valGround.max()
                                                     valNrmse / (valOut.numel()) ** (1/2))))

                    valPSNR_avg = (20 *
                                    torch.log10(1 / (torch.norm((modelExpected-valOut).reshape(modelExpected.shape[0], -1), dim = (1)).squeeze() / (valOut[0,0].numel()) ** (1/2))))

                    valLosses.append(valLoss)
                    valNrmses.append(valNrmse)
                    valPsnrs.append(valPSNR)
                    if wandbFlag:
                        wandbLoggerDict["valid_nRMSE"] = valNrmse
                        wandbLoggerDict["ref_nRMSE"] = torch.norm(theNoisyData-modelExpected)/torch.norm(modelExpected)
                        wandbLoggerDict["valid_pSNR"] = valPSNR
                        wandbLoggerDict["valid_pSNRavg"] = valPSNR_avg.mean(0)
                        wandbLoggerDict["valid_pSNRstd"] = valPSNR_avg.std(0)
                        wandbLoggerDict["valid_loss"] = valLoss

                print("---Epoch: {0}, Val Loss = {1:.6f}, Val nRMSE = {2:.6f}, Val pSNR = {3:.6f}".format(
                    epoch, valLoss, valNrmse, valPSNR))
        if wandbFlag:
            wandb.log(wandbLoggerDict)
    
    torch.save(model.state_dict(), saveDirectory+r"/" +
               "epoch" + str(epoch + 1) + "END.pth")
    return model, [trainLosses.numpy(), trainNrmses.numpy(), trainPsnrs.numpy()], [np.array(valLosses), np.array(valNrmses), np.array(valPsnrs)]

