
import torch
import torch.nn.functional as F
import numpy as np

def psnrForART(original,recon):
    return 10*torch.log10(original.shape[1]*(original.max())**2 / (torch.norm(original-recon, dim = 1))**2).cpu().numpy()

def ART(A, b, maxIter = 5, lambdaVal = 1, order = None, energy = None, underlyingImage = None):
    M, N = A.shape
    Nbatch = b.shape[0]
    x = torch.zeros((Nbatch, N), device=A.device, dtype = A.dtype)
    sqLambda = lambdaVal ** (1/2)
    residual = torch.zeros_like(b)
    
    
    if underlyingImage is not None:
        diagnoseVals = np.zeros(maxIter)
    
    if energy is None:
        energy = torch.norm(A, dim = (1))

    for ii in range(maxIter):
        for row in range(M):
            if order is not None:
                k = order[row]
            else:
                k = row
            beta = (b[:,k] - F.linear(x, A[k,:]) - sqLambda * residual[:, k]) / (energy[k]**2 + lambdaVal)
            x = x + beta[:,None] * A[k, :].repeat(Nbatch,1)
            residual[:, k] = residual[:, k] + beta * sqLambda
        x[x < 0] = 0
        
        if underlyingImage is not None:
            diagnoseVals[ii] = psnrForART(underlyingImage.reshape(underlyingImage.shape[0], -1) , x.detach().reshape(underlyingImage.shape[0], -1)).mean()
            print("max: ", ii, " lambda: ", lambdaVal , " pSNR:",  diagnoseVals[ii] )

    if underlyingImage is not None:
        return x, diagnoseVals
    else:
        return x


##!--- plug & play utils

class ppFnc3dnm():
    def __init__(self, threshold, model, imgShape = [32, 32]):
        self.tau = threshold
        self.model = model
        self.imgShape = imgShape

    def __call__(self,x):
        theShape = ((-1, 1, *self.imgShape))
        out = self.model(x.reshape(theShape)).reshape(x.shape)
        b = x-out
        bNorm = torch.norm(b)
        if bNorm <= self.tau:
            return out
        else:
            return out + ((1- self.tau/bNorm)*b).reshape(1, -1)

class ppFnc2dnm(): #both 2d and 3d without slice-batch
    def __init__(self,tau, model, imgShape = [32, 32], fwdBatch = None):
        self.tau = tau
        self.model = model
        self.imgShape = imgShape
        self.fwdBatch = fwdBatch
    def __call__(self,x):
        if not(self.fwdBatch is None):
            fwdBatch = self.fwdBatch
            self.fwdBatch = None
            xRet = torch.zeros_like(x)
            iii = 0
            while iii < x.shape[0]-(x.shape[0]%fwdBatch):
                xRet[iii:iii+fwdBatch] = self(x[iii:iii+fwdBatch])
                iii += fwdBatch
            xRet[iii:] = self(x[iii:])
            self.fwdBatch = fwdBatch
            return xRet
        else:
            theShape = ((-1, 1, *self.imgShape))
            xIn = x.reshape(theShape)
            return x-self.tau*(x-self.model(xIn).reshape(x.shape))
    
class ppFnc2dnmBatch3d():
    def __init__(self,tau, model, shape3d = [19, 19, 19], batchAxis = 2,fwdBatch = None):
        self.fwdBatch = fwdBatch
        self.tau = tau
        self.model = model
        self.shape3d = shape3d
        self.batchAxis = batchAxis
        # self.counterVerbose = 6
    def __call__(self,x):
        if not(self.fwdBatch is None):
            fwdBatch = self.fwdBatch
            self.fwdBatch = None
            xRet = torch.zeros_like(x)
            iii = 0
            while iii < x.shape[0]-(x.shape[0]%fwdBatch):
                xRet[iii:iii+fwdBatch] = self(x[iii:iii+fwdBatch])
                iii += fwdBatch
            xRet[iii:] = self(x[iii:])
            self.fwdBatch = fwdBatch
            return xRet
        else:
            shape3d = self.shape3d
            Nbatch = x.shape[0]
            if Nbatch == 1:
                if type(self.batchAxis) is int:
                    batchAxis = self.batchAxis
                    tIndcs = [0,1,2,3]
                    tIndcs.pop(batchAxis+1)
                    permFwd = (batchAxis+1,*tIndcs)
                    permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                    xIn = x.reshape(1,*self.shape3d)
                    x = x-self.tau*(x-(self.model(xIn.permute(*permFwd))).permute(*permBack).reshape(x.shape))

                else:
                    if -1 not in self.batchAxis: #Sequential
                        for batchAxis in self.batchAxis:
                            tIndcs = [0,1,2,3]
                            tIndcs.pop(batchAxis+1)
                            permFwd = (batchAxis+1,*tIndcs)
                            permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                            xIn = x.reshape(1,*self.shape3d)
                            x = x-self.tau*(x-(self.model(xIn.permute(*permFwd))).permute(*permBack).reshape(x.shape))
                    else:
                        xFin = torch.zeros_like(x)
                        for batchAxis in [x for x in self.batchAxis if x!=-1]:
                            tIndcs = [0,1,2,3]
                            tIndcs.pop(batchAxis+1)
                            permFwd = (batchAxis+1,*tIndcs)
                            permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                            xIn = x.reshape(1,*self.shape3d)
                            xFin += x-self.tau*(x-(self.model(xIn.permute(*permFwd))).permute(*permBack).reshape(x.shape))
                        x = xFin/(len(self.batchAxis)-1)
            else:
                if type(self.batchAxis) is int:
                    batchAxis = self.batchAxis
                    xIn = x.reshape(Nbatch,*self.shape3d)
                    tIndcs = [1,2,3]
                    tIndcs.pop(batchAxis)
                    permFwd = (0,batchAxis+1,*tIndcs)
                    permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                    xIn = xIn.permute(permFwd)
                    xIn = xIn.reshape(-1,1,*xIn.shape[2:])
                    xIn = self.model(xIn)
                    xIn = xIn.reshape(Nbatch,shape3d[batchAxis],*xIn.shape[-2:])
                    xIn = xIn.permute(permBack).reshape(x.shape)
                    x = x-self.tau*(x-xIn)
                else:
                    if -1 not in self.batchAxis: #Sequential
                        for batchAxis in self.batchAxis:
                            xIn = x.reshape(Nbatch,*self.shape3d)
                            tIndcs = [1,2,3]
                            tIndcs.pop(batchAxis)
                            permFwd = (0,batchAxis+1,*tIndcs)
                            permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                            xIn = xIn.permute(permFwd)
                            xIn = xIn.reshape(-1,1,*xIn.shape[2:])
                            xIn = self.model(xIn)
                            xIn = xIn.reshape(Nbatch,shape3d[batchAxis],*xIn.shape[-2:])
                            xIn = xIn.permute(permBack).reshape(x.shape)
                            x = x-self.tau*(x-xIn)
                    else:
                        xFin = torch.zeros_like(x)
                        for batchAxis in [x for x in self.batchAxis if x!=-1]:
                            xIn = x.reshape(Nbatch,*self.shape3d)
                            tIndcs = [1,2,3]
                            tIndcs.pop(batchAxis)
                            permFwd = (0,batchAxis+1,*tIndcs)
                            permBack = (permFwd.index(0),permFwd.index(1),permFwd.index(2),permFwd.index(3))
                            xIn = xIn.permute(permFwd)
                            xIn = xIn.reshape(-1,1,*xIn.shape[2:])
                            xIn = self.model(xIn)
                            xIn = xIn.reshape(Nbatch,shape3d[batchAxis],*xIn.shape[-2:])
                            xIn = xIn.permute(permBack).reshape(x.shape)
                            xFin += x-self.tau*(x-xIn)
                        x = xFin/(len(self.batchAxis)-1)
            return x    

class ppFnc2dnmSliceWise():
    def __init__(self,tau, model, shape3d = [19, 19, 19], batchAxis = 2,fwdBatch = None):
        self.fwdBatch = fwdBatch
        self.tau = tau
        self.model = model
        self.shape3d = shape3d
        self.batchAxis = batchAxis
        print(batchAxis)
        
    def __call__(self,x):
        # OpenMPI Run-time analiz için start
        shape3d = self.shape3d
        Nbatch = x.shape[0]
        if Nbatch == 1:
            if self.batchAxis == 0:
                xRes = x.reshape(self.shape3d[0], 1, self.shape3d[1], self.shape3d[2])
                xOut = (xRes-self.tau*(xRes-self.model(xRes))).squeeze()
                x = xOut.reshape(1, -1)
            else:
                xRes = x.reshape(*self.shape3d)
                xAllIn = torch.cat((xRes, xRes.permute((1, 0, 2)), xRes.permute((2, 0, 1))), dim = 0).reshape(self.shape3d[0] * 3, 1, self.shape3d[1], self.shape3d[2])
                xAllIn = (xAllIn-self.tau*(xAllIn-self.model(xAllIn))).squeeze()
                xOut = (xAllIn[0:self.shape3d[0]] + xAllIn[self.shape3d[0]:(self.shape3d[0]*2)].permute((1, 0, 2)) + xAllIn[(2*self.shape3d[0]):].permute((1, 2, 0))) / 3
                x = xOut.reshape(1, -1)
        return x

        
class ppFnc():
    def __init__(self,tau, model, imgShape = [32, 32]):
        self.tau = tau
        self.model = model
        self.imgShape = imgShape

    def __call__(self,x):
        theShape = ((-1, 1, *self.imgShape))
        return self.model(x.reshape(theShape)).reshape(x.shape)
    
    
def Q1(x):
    return x - x.roll(1,1)
def Q2(x):
    return x - x.roll(1,2)
def Q3(x):
    return x - x.roll(1,3)

class softTV():
    def __init__(self,tau, imgShape = [32, 32], numIter=30):
        self.tau = tau
        self.numIter = numIter
        self.imgShape = imgShape
        
    def __call__(self,x):
        if len(self.imgShape) == 2:
            theShape = ((-1, *self.imgShape))
            tau = self.tau 
            numIter = self.numIter         
            if (tau == 0):
                return x

            xShp = x.shape
            x = x.reshape(theShape)
            pn2 = pn1 = torch.zeros_like(x)
            for ii in range(numIter):
                myQst = (pn1.roll(-1,1) - pn1 + pn2.roll(-1,2) - pn2)
                vn1 = pn1 + 0.25 * Q1(myQst - x/tau)
                vn2 = pn2 + 0.25 * Q2(myQst - x/tau)
                d1 = vn1.abs()
                d2 = vn2.abs()
                pn1 = vn1 / (d1 * (d1 > 1) + (d1 <= 1))
                pn2 = vn2 / (d2 * (d2 > 1) + (d2 <= 1))
            return (x - tau * (pn1.roll(-1,1) - pn1 + pn2.roll(-1,2) - pn2)).reshape(xShp)
        elif len(self.imgShape) == 3:
            theShape = ((-1, *self.imgShape))
            tau = self.tau 
            numIter = self.numIter         
            if (tau == 0):
                return x

            xShp = x.shape
            x = x.reshape(theShape)
            pn3 = pn2 = pn1 = torch.zeros_like(x)
            for ii in range(numIter):
                myQst = (pn1.roll(-1,1) - pn1 + pn2.roll(-1,2) - pn2 + pn3.roll(-1,3) - pn3)
        #        myQst = Qstar(pn1, pn2)
                vn1 = pn1 + 1/6 * Q1(myQst - x/tau)
                vn2 = pn2 + 1/6 * Q2(myQst - x/tau)
                vn3 = pn3 + 1/6 * Q3(myQst - x/tau)
                d1 = vn1.abs()
                d2 = vn2.abs()
                d3 = vn3.abs()
                pn1 = vn1 / (d1 * (d1 > 1) + (d1 <= 1))
                pn2 = vn2 / (d2 * (d2 > 1) + (d2 <= 1))
                pn3 = vn3 / (d3 * (d3 > 1) + (d3 <= 1))
            return (x - tau * (pn1.roll(-1,1) - pn1 + pn2.roll(-1,2) - pn2+ pn3.roll(-1,3) - pn3)).reshape(xShp)

class softT():
    def __init__(self, tau):
        self.tau = tau
    def __call__(self,x):
        getSgn = torch.sgn(x)
        getMax = torch.abs(x) - self.tau
        getMax = getMax * (torch.sign(getMax) > 0) # enforce positivity
        return getSgn * getMax
        
class softTpos():
    def __init__(self, tau):
        self.tau = tau
    def __call__(self,x):
        getSgn = x > 0
        getMax = torch.abs(x) - self.tau
        getMax = getMax * (torch.sign(getMax) > 0) # enforce positivity
        return getSgn * getMax
        
def proj2Tmtx(s, y, epsilon):
    if isinstance(epsilon, torch.Tensor):
        nrmVal = torch.linalg.norm(s - y, dim=(1,2))
        theInd = nrmVal < epsilon.squeeze()
        nrmVal[theInd] = epsilon.squeeze()[theInd]
        
        return y + (epsilon / nrmVal[:,None,None]) * (s - y)
    else:
        nrmVal = torch.linalg.norm(s - y, dim=(1,2))
        nrmVal[nrmVal < epsilon] = epsilon
        return y + (epsilon / nrmVal[:,None,None]) * (s - y)

##!--- ADMM codes