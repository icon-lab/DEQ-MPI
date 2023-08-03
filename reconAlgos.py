import numpy as np
import torch
import torch.nn.functional as F
from modelClasses import * #DnCNNDenoiser, rdnDenoiserResRelu, rdnEnd2EndnsMultShared, rdnADMMnet
# from modelClasses import rdnADMMnet
import os
from reconUtils import *

# psnr ve hfen cell'i

class returnHfen():
    def __init__(self, imgSize):
        Nx = 7
        sig = 0.7
        xAxs = np.linspace(-3,3,Nx)
        xx, yy = np.meshgrid(xAxs, xAxs,indexing='ij')
        logFilter = -1/(np.pi*sig**4) * (1 - ((xx**2 + yy**2))/(2*sig**2)) * np.exp(-(xx**2 + yy**2) / (2 * sig**2))
        self.logFilterTorch = torch.from_numpy(logFilter).float().cuda().reshape(1,1,Nx,Nx)
        self.imgSize = imgSize

    def __call__(self, diff): # requires input of size .reshape(-1,1,32,32)
        return (torch.norm(torch.nn.functional.conv2d(diff.reshape((-1, 1, *self.imgSize)), self.logFilterTorch, padding='same'), dim = (2, 3))).squeeze().cpu().numpy()

def psnr(original,recon):
    return 10*torch.log10(original.shape[1]*(original.max())**2 / (torch.norm(original-recon, dim = 1))**2).cpu().numpy()
#      return 10*torch.log10(torch.numel(original)*(original.max())**2 / (torch.norm(original-recon))**2)    

def objectiveValues(x, imgSize = [32, 32]):
    theShape = ((-1, *imgSize))
    return (torch.sum(x.abs(), dim=1)).detach().cpu().numpy(), ( torch.sum( torch.sqrt(Q1(x.reshape(theShape))**2 + Q2(x.reshape(theShape))**2), dim = (1, 2) )   ).detach().cpu().numpy()

class ADMMfncs:
    def __init__(self, AtC, MaxIter, verboseIn, imgSize, diagnose = True):
        self.MaxIter = MaxIter
        self.imgSize = imgSize
        if diagnose == True:
            self.hfenFnc = returnHfen(imgSize) if len(imgSize) == 2 else returnHfen(imgSize[1:])
        else:
            self.hfenFnc = lambda x: 0
        self.verboseIn = verboseIn
        self.diagnose = diagnose

        if AtC is not None:
            self.updateForwardModel(AtC)

    def updateForwardModel(self, AtC):

        self.AtC = AtC
        self.AtCT = AtC.T

        n1 = self.imgSize[0]
        n2 = self.imgSize[1]
        n3 = 1 if len(self.imgSize) == 2 else self.imgSize[2]
        N = n1 * n2 *n3
        self.MtC = torch.inverse(torch.eye(N).type_as(self.AtCT) + self.AtCT @ self.AtC)
        self.MtC2 = torch.inverse(2 * torch.eye(N).type_as(self.AtCT) + self.AtCT @ self.AtC)

    def calculateDiagnoseVals(self, x, refVals):
        if self.diagnose == False:
            return 0,0,0,0,0,0
        outDiag = (torch.norm(x - refVals, dim = 1)).detach().cpu().numpy()
        outMax = (x.abs().max()).detach().cpu().numpy()
        outPsnrs = (psnr(refVals,x))
        if len(self.imgSize) == 2:
            outHfens = (self.hfenFnc(x-refVals)) if len(self.imgSize) == 2 else 0
            l1Vals, TVvals = objectiveValues(x.detach(), self.imgSize)
        else:
            outHfens = 0
            l1Vals, TVvals = 0, 0
            for ii in range(self.imgSize[0]):
                outHfens += (self.hfenFnc((x-refVals).detach().reshape(self.imgSize)[ii].reshape(-1, self.imgSize[1] * self.imgSize[2])))
                l1ValsT, TVvalsT = objectiveValues(x.detach().reshape(self.imgSize)[ii].reshape(-1, self.imgSize[1] * self.imgSize[2]), self.imgSize[1:])
                l1Vals += l1ValsT
                TVvals += TVvalsT
        
        return outDiag, outMax, outPsnrs, outHfens, l1Vals, TVvals

    def defineZeros(self, Nbatch):  
        MaxIter = self.MaxIter      
        outDiag = np.zeros((Nbatch, MaxIter))
        outMax = np.zeros((Nbatch, MaxIter))
        outPsnrs = np.zeros((Nbatch, MaxIter))
        outHfens = np.zeros((Nbatch, MaxIter))
        inpOutNorms = np.zeros((Nbatch, MaxIter))
        inpOutNrmses = np.zeros((Nbatch, MaxIter))

        l1Vals = np.zeros((Nbatch, MaxIter))
        TVvals = np.zeros((Nbatch, MaxIter))
        critVals = np.zeros((Nbatch, MaxIter))

        return outDiag, outMax, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, l1Vals, TVvals, critVals
    def afterFnc(self, datatC, fnc1, fnc2, after, epsilon, refVals, mu2Scale=1,muScaleIter=1, x_in = None):
        x_conv = F.linear(datatC.squeeze(), self.AtCT)

        Nbatch = datatC.shape[0]
        
        d0 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC)
        if (x_in is None):
            z0 = torch.zeros_like(x_conv, )
            z2 = torch.zeros_like(datatC)
        else:
            z0 = x_in.reshape(x_conv.shape)
            z2 = F.linear(z0, self.AtC)
        
        if self.diagnose:
            refNrm = float(torch.norm(refVals))
            refHfens = self.hfenFnc(refVals) if len(self.imgSize) == 2 else 0

            outDiag, outMax, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, l1Vals, TVvals, critVals = self.defineZeros(Nbatch)
        
        for ii in range(self.MaxIter):
            r = (z0 + d0) + F.linear(z2 + d2, self.AtCT)
            x = F.linear(r, self.MtC)
            Ax = F.linear(x, self.AtC)
            
            if (ii > after):
                z0 = fnc2(x - d0) 
            else:
                z0 = fnc1(x - d0)

            if self.diagnose:
                inpOutNorms[:,ii] = (torch.norm(x - d0-z0, dim = 1)).detach().cpu().numpy()
                inpOutNrmses[:,ii] = inpOutNorms[:, ii]/(torch.norm(x-d0, dim = 1)).detach().cpu().numpy()

            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2
                
            if (ii+1)%muScaleIter==0:    
                fnc1.tau /= mu2Scale
                fnc2.tau /= mu2Scale
                d0 /= mu2Scale
                d2 /= mu2Scale

            if self.diagnose:
                outDiag[:,ii], outMax[:, ii], outPsnrs[:, ii], outHfens[:, ii], l1Vals[:, ii], TVvals[:, ii] = self.calculateDiagnoseVals(x, refVals)
                outDiag[:,ii] /= refNrm
                outHfens[:,ii] /= refHfens
                epsilonTmp = epsilon.cpu().numpy() if isinstance(epsilon, torch.Tensor) else epsilon
                critVals[:, ii] = torch.sqrt(torch.sum((Ax - datatC).reshape(Nbatch, -1).abs() ** 2, dim = 1)).detach().cpu().numpy() / epsilonTmp
            
                if ii % self.verboseIn == 1:
                    farkGraph = outDiag[:,ii]
                    print(str(ii) + ' - cost: {0:.5f} & {1:.5f}'.format(l1Vals[0, ii], TVvals[0, ii]) + ' crit: {0:.5f}'.format(critVals[0, ii]) + ' / nRMSE: {0}'.format(farkGraph))
        if self.diagnose:
            return x, outDiag, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, outMax, l1Vals, TVvals, critVals
        else:
            return x, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def afterFncLD(self, datatC, fnc1, fnc2, after, epsilon, refVals, mu2Scale=1,muScaleIter=1, consistencyFnc = lambda s, y, epsilon, Ua: proj2Tmtx(s, y, epsilon), x_in = None):
        x_conv = F.linear(datatC.squeeze(), self.AtCT)

        Nbatch = datatC.shape[0]
        
        d0 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC)
        if (x_in is None):
            z0 = torch.zeros_like(x_conv, )
            z2 = torch.zeros_like(datatC)
        else:
            z0 = x_in.reshape(x_conv.shape)
            z2 = F.linear(z0, self.AtC)
        
        if self.diagnose:
            refNrm = float(torch.norm(refVals))
            refHfens = self.hfenFnc(refVals) if len(self.imgSize) == 2 else 0

            outDiag, outMax, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, l1Vals, TVvals, critVals = self.defineZeros(Nbatch)
        
        for ii in range(self.MaxIter):
            r = (z0 + d0) + F.linear(z2 + d2, self.AtCT)
            x = F.linear(r, self.MtC)
            Ax = F.linear(x, self.AtC)
            
            if (ii > after):
                z0 = fnc2(x - d0) 
            else:
                z0 = fnc1(x - d0)

            if self.diagnose:
                inpOutNorms[:,ii] = (torch.norm(x - d0-z0, dim = 1)).detach().cpu().numpy()
                inpOutNrmses[:,ii] = inpOutNorms[:, ii]/(torch.norm(x-d0, dim = 1)).detach().cpu().numpy()

            z2 = consistencyFnc((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon, 1).reshape(Nbatch, -1)

            d0 = d0 - x + z0
            d2 = d2 - Ax + z2
                
            if (ii+1)%muScaleIter==0:    
                fnc1.tau /= mu2Scale
                fnc2.tau /= mu2Scale
                d0 /= mu2Scale
                d2 /= mu2Scale

            if self.diagnose:
                outDiag[:,ii], outMax[:, ii], outPsnrs[:, ii], outHfens[:, ii], l1Vals[:, ii], TVvals[:, ii] = self.calculateDiagnoseVals(x, refVals)
                outDiag[:,ii] /= refNrm
                outHfens[:,ii] /= refHfens
                epsilonTmp = epsilon.cpu().numpy() if isinstance(epsilon, torch.Tensor) else epsilon
                critVals[:, ii] = torch.sqrt(torch.sum((Ax - datatC).reshape(Nbatch, -1).abs() ** 2, dim = 1)).detach().cpu().numpy() / epsilonTmp
            
                if ii % self.verboseIn == 1:
                    farkGraph = outDiag[:,ii]
                    print(str(ii) + ' - cost: {0:.5f} & {1:.5f}'.format(l1Vals[0, ii], TVvals[0, ii]) + ' crit: {0:.5f}'.format(critVals[0, ii]) + ' / nRMSE: {0}'.format(farkGraph))
        if self.diagnose:
            return x, outDiag, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, outMax, l1Vals, TVvals, critVals
        else:
            return x, 0, 0, 0, 0, 0, 0, 0, 0, 0
    

    def ADMMreconDual(self, datatC, fnc1,fnc2, epsilon, refVals, mu2Scale=1,muScaleIter=1, x_in = None):
        # initialize vectors
        x_conv = F.linear(datatC, self.AtCT)
        
        Nbatch = datatC.shape[0]


        d0 = torch.zeros_like(x_conv, )
        d1 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC)
        if (x_in is None):
            z0 = torch.zeros_like(x_conv, )
            z1 = torch.zeros_like(x_conv, )
            z2 = torch.zeros_like(datatC)
        else:
            z0 = x_in.reshape(x_conv.shape)
            z1 = x_in.reshape(x_conv.shape)
            z2 = F.linear(z0, self.AtC)
        if self.diagnose:
            refNrm = float(torch.norm(refVals))
            refHfens = self.hfenFnc(refVals) if len(self.imgSize) == 2 else 0

            outDiag, outMax, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, l1Vals, TVvals, critVals = self.defineZeros(Nbatch)
        
        for ii in range(self.MaxIter):
            r = (z0 + d0) + (z1 + d1) + F.linear(z2 + d2, self.AtCT)
            x = F.linear(r, self.MtC2)
            Ax = F.linear(x, self.AtC)
            
            z0 = fnc1(x - d0)
            if self.diagnose:
                inpOutNorms[:,ii] = (torch.norm(x - d0-z0, dim = 1)).detach().cpu().numpy()
                inpOutNrmses[:,ii] = inpOutNorms[:, ii]/(torch.norm(x-d0, dim = 1)).detach().cpu().numpy()

            z1 = fnc2((x - d1))
            
            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1) #...hape(Nbatch*2, 1, -1) 

            d0 = d0 - x + z0
            d1 = d1 - x + z1
            d2 = d2 - Ax + z2
            
            if (ii+1)%muScaleIter==0:
                # print('halved tau at ', ii)
                fnc1.tau /= mu2Scale
                fnc2.tau /= mu2Scale
                d0 /= mu2Scale
                d1 /= mu2Scale
                d2 /= mu2Scale
                
            if self.diagnose:

                outDiag[:,ii], outMax[:, ii], outPsnrs[:, ii], outHfens[:, ii], l1Vals[:, ii], TVvals[:, ii] = self.calculateDiagnoseVals(x, refVals)
                outDiag[:,ii] /= refNrm
                outHfens[:,ii] /= refHfens
                critVals[:, ii] = torch.sqrt(torch.sum((Ax - datatC).reshape(Nbatch, -1).abs() ** 2, dim = 1)).detach().cpu().numpy() / epsilon

                if ii % self.verboseIn == 1:
                    farkGraph = outDiag[:,ii]
                    print(str(ii) + ' - cost: {0:.5f} & {1:.5f}'.format(l1Vals[0, ii], TVvals[0, ii]) + ' crit: {0:.5f}'.format(critVals[0, ii]) + ' / nRMSE: {0}'.format(farkGraph))
        if self.diagnose:
            return x, outDiag, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, outMax, l1Vals, TVvals, critVals
        else:
            return x, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def ADMMreconDualSynthesis(self, datatC, fnc1, fnc2, epsilon, refVals, mu2Scale=1,muScaleIter=1, x_in = None):
        # initialize vectors
        x_conv = F.linear(datatC, self.AtCT)
        
        # Acat = torch.cat((self.AtC, self.AtC), dim = 1)

        Nbatch = datatC.shape[0]

        tmpC = torch.inverse(torch.eye(self.AtC.shape[0]).type_as(self.AtCT) + 2 * self.AtC @ self.AtCT)
        
        d0 = torch.zeros_like(x_conv, )
        d1 = torch.zeros_like(x_conv, )
        d2 = torch.zeros_like(datatC)
        if (x_in is None):
            z0 = torch.zeros_like(x_conv, )
            z1 = torch.zeros_like(x_conv, )
            z2 = torch.zeros_like(datatC)
        else:
            z0 = x_in.reshape(x_conv.shape) / 2
            z1 = x_in.reshape(x_conv.shape) / 2
            z2 = F.linear(z0 + z1, self.AtC)
        
        refNrm = float(torch.norm(refVals))
        refHfens = self.hfenFnc(refVals) if len(self.imgSize) == 2 else 0

        outDiag, outMax, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, l1Vals, TVvals, critVals = self.defineZeros(Nbatch)

        
        for ii in range(self.MaxIter):
            
            qVal = F.linear(z2 + d2, self.AtCT)
            
            r1 = (z0 + d0) + qVal
            r2 = (z1 + d1) + qVal
            
            tmpVal = F.linear(F.linear(F.linear(r1 + r2, self.AtC), tmpC), self.AtCT)
            
            x1 = r1 - tmpVal
            x2 = r2 - tmpVal
                    
            Ax = F.linear(x1 + x2, self.AtC)
            
            z0 = fnc1(x1 - d0)
            
            inpOutNorms[:,ii] = (torch.norm(x1 - d0-z0, dim = 1)).detach().cpu().numpy()
            inpOutNrmses[:,ii] = inpOutNorms[:, ii]/(torch.norm(x1-d0, dim = 1)).detach().cpu().numpy()

            z1 = fnc2((x2 - d1))
            
            z2 = proj2Tmtx((Ax - d2).reshape(Nbatch, 1, -1),
                        datatC.reshape(Nbatch, 1, -1), epsilon).reshape(Nbatch, -1) #...hape(Nbatch*2, 1, -1) 

            d0 = d0 - x1 + z0
            d1 = d1 - x2 + z1
            d2 = d2 - Ax + z2
            
            if (ii+1)%muScaleIter==0:
                # print('halved tau at ', ii)
                fnc1.tau /= mu2Scale
                fnc2.tau /= mu2Scale
                d0 /= mu2Scale
                d1 /= mu2Scale
                d2 /= mu2Scale
                
            x = x1 + x2

            outDiag[:,ii], outMax[:, ii], outPsnrs[:, ii], outHfens[:, ii], l1Vals[:, ii], TVvals[:, ii]= self.calculateDiagnoseVals(x, refVals)
            outDiag[:,ii] /= refNrm
            outHfens[:,ii] /= refHfens
            critVals[:, ii] = torch.sqrt(torch.sum((Ax - datatC).reshape(Nbatch, -1).abs() ** 2, dim = 1)).detach().cpu().numpy() / epsilon
            
            if ii % self.verboseIn == 1:
                farkGraph = outDiag[:,ii]
                print(str(ii) + ' - cost: {0:.5f} & {1:.5f}'.format(l1Vals[0, ii], TVvals[0, ii]) + ' crit: {0:.5f}'.format(critVals[0, ii]) + ' / nRMSE: {0}'.format(farkGraph))

        return x, outDiag, outPsnrs, outHfens, inpOutNorms, inpOutNrmses, outMax, l1Vals, TVvals, critVals, x1, x2